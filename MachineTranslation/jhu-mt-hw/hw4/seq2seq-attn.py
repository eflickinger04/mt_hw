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
#if you are running on the gradx/ugradx/ another cluster, 
#you will need the following line
#if you run on a local machine, you can comment it out
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
from torch.utils.data import Dataset, DataLoader



logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
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
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
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
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################

from torch.nn.utils.rnn import pad_sequence


def prepare_batch(pairs, src_vocab, tgt_vocab):
    input_tensors = []
    target_tensors = []
    for pair in pairs:
        input_tensor, target_tensor = tensors_from_pair(src_vocab, tgt_vocab, pair)
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)

    # compute lengths
    input_lengths = torch.tensor([len(t) for t in input_tensors], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in target_tensors], dtype=torch.long)

    # pad sequences to the max length 
    input_batch = pad_sequence(input_tensors, batch_first=True, padding_value=EOS_index)
    target_batch = pad_sequence(target_tensors, batch_first=True, padding_value=EOS_index)

    # sort the batch by input_lengths
    input_lengths, perm_idx = input_lengths.sort(0, descending=True)
    input_batch = input_batch[perm_idx]
    target_batch = target_batch[perm_idx]
    target_lengths = target_lengths[perm_idx]

    return input_batch, target_batch, input_lengths, target_lengths



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True) 

    def forward(self, input_tensor, input_lengths):
        # input_tensor: batch_size x seq_length
        embedded = self.embedding(input_tensor)  # batch_size x seq_length x hidden_size
        # pack padded seq
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.lstm(packed)
        # pack seq
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        # Define the attention model
        if self.attn_model == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.attn_model == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))
        # No additional parameters for 'dot' model

        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)  # Adjusted input size for input-feeding
        self.out = nn.Linear(hidden_size, output_size)

    def compute_score(self, hidden, encoder_outputs):
        """
        Compute attention energies between decoder hidden state and encoder outputs.

        Args:
            hidden: (batch_size, hidden_size)
            encoder_outputs: (batch_size, seq_len, hidden_size)
        Returns:
            attn_energies: (batch_size, seq_len)
        """
        if self.attn_model == 'dot':
            # For dot, compute dot product
            attn_energies = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
        elif self.attn_model == 'general':
            # For general, apply linear layer to encoder outputs
            attn_energies = torch.bmm(self.attn(encoder_outputs), hidden.unsqueeze(2)).squeeze(2)
        elif self.attn_model == 'concat':
            seq_len = encoder_outputs.size(1)
            hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_size)
            concat = torch.cat((hidden_expanded, encoder_outputs), dim=2)  # (batch_size, seq_len, hidden_size * 2)
            attn_energies = self.attn(concat)  # (batch_size, seq_len, hidden_size)
            attn_energies = torch.tanh(attn_energies)  # (batch_size, seq_len, hidden_size)
            attn_energies = torch.sum(self.v * attn_energies, dim=2)  # (batch_size, seq_len)
        else:
            raise NotImplementedError

        return attn_energies  # (batch_size, seq_len)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs, last_attn_hidden):
        """
        Args:
            decoder_input: Tensor of shape (batch_size, 1)
            decoder_hidden: Tuple (h_n, c_n), each of shape (num_layers, batch_size, hidden_size)
            encoder_outputs: Tensor of shape (batch_size, seq_len, hidden_size)
            last_attn_hidden: Tensor of shape (batch_size, 1, hidden_size)
        Returns:
            output: Tensor of shape (batch_size, output_size)
            decoder_hidden: Updated hidden state tuple
            attn_weights: Tensor of shape (batch_size, seq_len)
            attn_hidden: Attentional hidden state to be fed into next time step
        """
        batch_size = decoder_input.size(0)
        embedded = self.embedding(decoder_input)  # (batch_size, 1, hidden_size)
        embedded = self.dropout(embedded)

        # Get the last layer's hidden state
        hidden_h = decoder_hidden[0][-1]  # (batch_size, hidden_size)

        # Compute attention energies
        attn_energies = self.compute_score(hidden_h, encoder_outputs)  # (batch_size, seq_len)

        # Compute attention weights
        attn_weights = F.softmax(attn_energies, dim=1)  # (batch_size, seq_len)

        # Compute context vector as the weighted sum of encoder outputs
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch_size, 1, hidden_size)

        # Attentional hidden state
        attn_hidden = torch.tanh(self.attn_combine(torch.cat((embedded, context), dim=2)))  # (batch_size, 1, hidden_size)

        # Input to LSTM is concatenation of embedded input and last_attn_hidden
        lstm_input = torch.cat((embedded, last_attn_hidden), dim=2)  # (batch_size, 1, hidden_size * 2)

        output, decoder_hidden = self.lstm(lstm_input, decoder_hidden)  # output: (batch_size, 1, hidden_size)

        output = F.log_softmax(self.out(output.squeeze(1)), dim=1)  # (batch_size, output_size)

        return output, decoder_hidden, attn_weights.squeeze(1), attn_hidden


######################################################################

def train(input_tensor, target_tensor, input_lengths, target_lengths, encoder, decoder, optimizer, criterion):
    batch_size = input_tensor.size(0)

    encoder.train()
    decoder.train()

    optimizer.zero_grad()
    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)
    # encoder_outputs: (batch_size, seq_len, hidden_size)

    decoder_input = torch.tensor([SOS_index] * batch_size, device=device).unsqueeze(1)  # (batch_size, 1)
    decoder_hidden = encoder_hidden

    max_target_length = target_tensor.size(1)

    use_teacher_forcing = random.random() < 0.5

    # Initialize last_attn_hidden
    last_attn_hidden = torch.zeros(batch_size, 1, decoder.hidden_size, device=device)

    if use_teacher_forcing:
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention, attn_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_attn_hidden)
            target = target_tensor[:, di]  # (batch_size)
            loss += criterion(decoder_output, target)
            decoder_input = target.unsqueeze(1)  # (batch_size, 1)
            # Update last_attn_hidden
            last_attn_hidden = attn_hidden.detach()
    else:
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention, attn_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_attn_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # (batch_size, 1)
            target = target_tensor[:, di]  # (batch_size)
            loss += criterion(decoder_output, target)
            # Update last_attn_hidden
            last_attn_hidden = attn_hidden.detach()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item() / batch_size

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    Translates a single sentence using the encoder and decoder models.
    """
    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Prepare input tensor
        input_tensor = tensor_from_sentence(src_vocab, sentence).unsqueeze(0)  # Shape: (1, seq_len)
        input_lengths = torch.tensor([input_tensor.size(1)], dtype=torch.long, device=device)  # Shape: (1)

        # Move tensors to device
        input_tensor = input_tensor.to(device)
        input_lengths = input_lengths.to(device)

        # Encode the input sentence
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths)

        # Initialize decoder input and hidden state
        decoder_input = torch.tensor([SOS_index], device=device).unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, 1)
        decoder_hidden = encoder_hidden

        # Initialize last_attn_hidden
        last_attn_hidden = torch.zeros(1, 1, decoder.hidden_size, device=device)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, encoder_outputs.size(1), device=device)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention, attn_hidden = decoder(
                decoder_input.squeeze(1), decoder_hidden, encoder_outputs, last_attn_hidden)
            decoder_attentions[di] = decoder_attention[0]

            # Update last_attn_hidden
            last_attn_hidden = attn_hidden.detach()

            # Get top prediction
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word.get(topi.item(), '<UNK>'))

            # Prepare next input
            decoder_input = topi.detach().unsqueeze(1)  # Shape: (1, 1)

        return decoded_words, decoder_attentions[:di + 1]

######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
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
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    
    "*** YOUR CODE HERE ***"
    fig, ax = plt.subplots(figsize=(10, 8)) 
    
    cax = ax.matshow(attentions.numpy(), cmap='viridis')
    
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    fig.colorbar(cax)
    plt.xlabel('Source Sentence')
    plt.ylabel('Target Translation')
    
    plt.title('Attention Heatmap') 
    plt.savefig('attention_plot.png')
    plt.show()  # Display the plot
    #raise NotImplementedError


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
    ap.add_argument('--n_iters', default=150000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=1000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
    ap.add_argument('--attn_model', default='concat', choices=['dot', 'general', 'concat'],
                    help='Attention model to use')
    ap.add_argument('--num_epochs', default=10, type=int,
                    help='Number of epochs to train')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.0006, type=int,
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

 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iteration = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.attn_model, args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)


    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss(ignore_index=EOS_index)

    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    # prep batches
    batch_size = args.batch_size
    num_batches = len(train_pairs) // batch_size

    start = time.time()
    print_loss_total = 0  
    iteration = 0

    for epoch in range(args.num_epochs):
        random.shuffle(train_pairs) 
        for i in range(0, len(train_pairs), batch_size):
            iteration += batch_size
            batch_pairs = train_pairs[i:i+batch_size]
            input_batch, target_batch, input_lengths, target_lengths = prepare_batch(batch_pairs, src_vocab, tgt_vocab)

            # sort batch in desc order of input_lengths (required for pack_padded_sequence)
            input_lengths, perm_idx = (input_lengths).sort(0, descending=True)
            input_batch = input_batch[perm_idx]
            target_batch = target_batch[perm_idx]
            target_lengths = [target_lengths[j] for j in perm_idx]

            loss = train(input_batch, target_batch, input_lengths, target_lengths, encoder,
                         decoder, optimizer, criterion)
            print_loss_total += loss

            if iteration % args.checkpoint_every == 0:
                state = {'iter_num': iteration,
                         'enc_state': encoder.state_dict(),
                         'dec_state': decoder.state_dict(),
                         'opt_state': optimizer.state_dict(),
                         'src_vocab': src_vocab,
                         'tgt_vocab': tgt_vocab,
                         }
                filename = 'state_%010d.pt' % iteration
                torch.save(state, filename)
                logging.debug('wrote checkpoint to %s', filename)

            if iteration % args.print_every == 0:
                print_loss_avg = print_loss_total / args.print_every
                print_loss_total = 0
                elapsed_time = time.time() - start
                logging.info('Time since start: %.2f sec (iter:%d) loss_avg:%.4f',
                             elapsed_time,
                             iteration,
                             print_loss_avg)
                # translate
                translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
                translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

                references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
                candidates = [clean(sent).split() for sent in translated_sentences]
                dev_bleu = corpus_bleu(references, candidates)
                logging.info('Dev BLEU score: %.2f', dev_bleu)

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