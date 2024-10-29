#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib
# if you are running on the gradx/ugradx/ another cluster, 
# you will need the following line
# if you run on a local machine, you can comment it out
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to conflict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

PAD_token = 0
SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 1
EOS_index = 2
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indices
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "<PAD>", SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 3  # Count PAD, SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.strip().split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.strip().split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the languages based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.strip().split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device)

def batchify(pairs, src_vocab, tgt_vocab):
    src_tensors = [tensor_from_sentence(src_vocab, pair[0]) for pair in pairs]
    tgt_tensors = [tensor_from_sentence(tgt_vocab, pair[1]) for pair in pairs]
    return src_tensors, tgt_tensors

def get_batches(src_tensors, tgt_tensors, batch_size):
    batches = []
    for i in range(0, len(src_tensors), batch_size):
        src_batch = src_tensors[i:i+batch_size]
        tgt_batch = tgt_tensors[i:i+batch_size]
        # Get lengths
        src_lengths = [len(seq) for seq in src_batch]
        tgt_lengths = [len(seq) for seq in tgt_batch]
        # Sort by src_lengths in decreasing order
        sorted_data = sorted(zip(src_lengths, src_batch, tgt_batch, tgt_lengths), key=lambda x: x[0], reverse=True)
        src_lengths, src_batch, tgt_batch, tgt_lengths = zip(*sorted_data)
        src_batch = list(src_batch)
        tgt_batch = list(tgt_batch)
        # Pad sequences
        src_padded = pad_sequence(src_batch, padding_value=PAD_token)
        tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_token)
        batches.append((src_padded, tgt_padded, src_lengths, tgt_lengths))
    return batches

######################################################################

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional)

    def forward(self, input_seq, input_lengths, hidden):
        # input_seq: (seq_len, batch_size)
        embedded = self.embedding(input_seq)  # (seq_len, batch_size, hidden_size)
        packed = pack_padded_sequence(embedded, input_lengths, enforce_sorted=True)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, _ = pad_packed_sequence(outputs)  # outputs: (seq_len, batch_size, hidden_size * num_directions)
        # Sum bidirectional outputs
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
            hidden = (hidden[0][::2] + hidden[0][1::2],
                      hidden[1][::2] + hidden[1][1::2])
        return outputs, hidden

    def get_initial_hidden_state(self, batch_size):
        num_directions = 2 if self.bidirectional else 1
        return (torch.zeros(num_directions, batch_size, self.hidden_size, device=device),
                torch.zeros(num_directions, batch_size, self.hidden_size, device=device))

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, bidirectional=True):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs, encoder_output_lengths):
        # input: (batch_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # encoder_outputs: (seq_len, batch_size, hidden_size)
        # encoder_output_lengths: list of lengths

        embedded = self.embedding(input).unsqueeze(0)  # (1, batch_size, hidden_size)
        embedded = self.dropout(embedded)

        decoder_hidden_state = hidden[0][-1]  # (batch_size, hidden_size)

        # Compute attention energies
        # encoder_outputs: (seq_len, batch_size, hidden_size)
        # Transpose to (batch_size, seq_len, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # (batch_size, seq_len, hidden_size)

        # Expand decoder hidden state
        decoder_hidden_state = decoder_hidden_state.unsqueeze(2)  # (batch_size, hidden_size, 1)

        # Compute dot product between encoder outputs and decoder hidden state
        attn_energies = torch.bmm(encoder_outputs, decoder_hidden_state).squeeze(2)  # (batch_size, seq_len)

        # Create mask based on encoder_output_lengths
        max_len = encoder_outputs.size(1)
        mask = torch.arange(max_len).unsqueeze(0).to(device)
        mask = mask >= torch.tensor(encoder_output_lengths).unsqueeze(1).to(device)

        attn_energies.data.masked_fill_(mask, -float('inf'))

        attn_weights = F.softmax(attn_energies, dim=1)  # (batch_size, seq_len)

        # Compute context vector
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_size)
        attn_applied = attn_applied.transpose(0, 1)  # (1, batch_size, hidden_size)

        output = torch.cat((embedded, attn_applied), 2)  # (1, batch_size, hidden_size * 2)
        output = F.relu(self.attn_combine(output[0])).unsqueeze(0)  # (1, batch_size, hidden_size)

        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)  # (batch_size, output_size)
        return output, hidden, attn_weights

    def get_initial_hidden_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))

######################################################################

def train(input_tensor_batch, target_tensor_batch, input_lengths, target_lengths, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):
    batch_size = input_tensor_batch.size(1)

    encoder_hidden = encoder.get_initial_hidden_state(batch_size)

    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor_batch, input_lengths, encoder_hidden)

    decoder_input = torch.tensor([SOS_index] * batch_size, device=device)  # (batch_size)

    decoder_hidden = encoder_hidden  # Use encoder's last hidden state as initial hidden state for decoder

    max_target_length = max(target_lengths)

    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size, device=device)

    use_teacher_forcing = True if random.random() < 0.5 else False

    if use_teacher_forcing:
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, input_lengths)
            all_decoder_outputs[t] = decoder_output
            if t < target_tensor_batch.size(0):
                decoder_input = target_tensor_batch[t]  # Next input is current target
            else:
                decoder_input = torch.tensor([PAD_token] * batch_size, device=device)
    else:
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, input_lengths)
            all_decoder_outputs[t] = decoder_output
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

    # Loss calculation with masking
    loss = 0
    for t in range(max_target_length):
        loss += criterion(all_decoder_outputs[t], target_tensor_batch[t])

    loss.backward()

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

    optimizer.step()

    return loss.item() / max_target_length

######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs translation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        input_lengths = [input_length]
        encoder_hidden = encoder.get_initial_hidden_state(1)

        encoder_outputs, encoder_hidden = encoder(input_tensor.unsqueeze(1), input_lengths, encoder_hidden)

        decoder_input = torch.tensor([SOS_index], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, input_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, input_lengths)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their translations
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences

######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgments:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

######################################################################

def show_attention(input_sentence, output_words, attentions):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    Your plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    
    # Prepare data for plotting
    fig, ax = plt.subplots(figsize=(10, 8)) 
    
    cax = ax.matshow(attentions.numpy(), cmap='viridis')
    
    ax.set_xticklabels([''] + input_sentence.strip().split(' '), rotation=90)
    ax.set_yticklabels([''] + output_words)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    fig.colorbar(cax)
    plt.xlabel('Source Sentence')
    plt.ylabel('Target Translation')
    
    plt.title('Attention Heatmap') 
    plt.savefig('attention_plot.png')
    plt.show()  # Display the plot

def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)

def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())

######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=75000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.0006, type=float,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size, bidirectional=True).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1, bidirectional=True).to(device)

    # encoder/decoder weights are randomly initialized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in data files
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # Prepare batches
    batch_size = 32  # You can adjust the batch size
    train_src_tensors, train_tgt_tensors = batchify(train_pairs, src_vocab, tgt_vocab)
    train_batches = get_batches(train_src_tensors, train_tgt_tensors, batch_size)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    total_batches = len(train_batches)
    n_epochs = args.n_iters // total_batches + 1

    for epoch in range(n_epochs):
        random.shuffle(train_batches)
        for batch in train_batches:
            iter_num += 1
            input_tensor_batch, target_tensor_batch, input_lengths, target_lengths = batch
            loss = train(input_tensor_batch, target_tensor_batch, input_lengths, target_lengths,
                         encoder, decoder, optimizer, criterion)
            print_loss_total += loss

            if iter_num % args.checkpoint_every == 0:
                state = {'iter_num': iter_num,
                         'enc_state': encoder.state_dict(),
                         'dec_state': decoder.state_dict(),
                         'opt_state': optimizer.state_dict(),
                         'src_vocab': src_vocab,
                         'tgt_vocab': tgt_vocab,
                         }
                filename = 'state_%010d.pt' % iter_num
                torch.save(state, filename)
                logging.debug('wrote checkpoint to %s', filename)

            if iter_num % args.print_every == 0:
                print_loss_avg = print_loss_total / args.print_every
                print_loss_total = 0
                logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                             time.time() - start,
                             iter_num,
                             iter_num / args.n_iters * 100,
                             print_loss_avg)
                # translate from the dev set
                translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
                translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

                references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
                candidates = [clean(sent).split() for sent in translated_sentences]
                dev_bleu = corpus_bleu(references, candidates)
                logging.info('Dev BLEU score: %.2f', dev_bleu)

            if iter_num >= args.n_iters:
                break
        if iter_num >= args.n_iters:
            break

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
