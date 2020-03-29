import torch
import torch.nn as nn

from .attention import AdditiveAttention

import torch.nn.functional as F

# TODO(joel): Encoder should return encoder_output and features map.
class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, vocab_size, encoder_size, hidden_size, attention_size):
        super(DecoderWithAttention, self).__init__()

        self.hidden_size = hidden_size

        self.attention = AdditiveAttention(encoder_size, hidden_size, attention_size)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def trainable_parameters(self):
        return list(self.parameters())

    def init_hidden(self, features):
        # Unsqueeze first dimension of features to apply num_layers = 1.
        # Shape after unsqueeze:
        #    features -- (num_layers(1), batch_size, hidden_size).
        # FIXME: LSTMcell no num_layers!!!!!!!!!!!!!
        h, c = torch.zeros(features.shape).to('cuda'), features
        return h, c

    def forward_step(self, features, x, state=None):
        """
        Forward decoder.

        features -- (batch_size(1), num_pixels(49), encoder_size(512)) -- encoder out
        x shape (1, )  # last prediction word.
        state: h -- (batch_size, hidden_size)
               c -- (batch_size, hidden_size)

        In training mode we use teacher forcing as we know the targets.
        """
        print(' ===== Prediction step =====')
        # ----------
        # Embeddings
        # ----------
        # captions -- (batch_size, embed_size)
        x = self.embedding(x)
        print ('Caption shape {}'.format(x.shape))


        #
        # initial state
        #
        if state is None:
            flat_features = features.view(features.size(0), -1)  # (batch_size, 7*7*encoder_size)
            fc1 = nn.Linear(flat_features.size(1), self.hidden_size).to('cuda')
            hidden = fc1(flat_features)
            
            state = self.init_hidden(hidden)
        else:
            hidden = state[0]

        print('Shape hidden: {}'.format(hidden.shape))
        print('Shape state: {}'.format(state[1].shape))

        # -------------
        # Decoder steps
        # -------------
        context, alphas = self.attention(features, hidden)
        print('Attention--> Context: {}, alphas: {}'.format(context.shape, alphas.shape))

        # ---------
        # LSTM step
        # ---------
        # hidden - (batch_size, hidden_size)
        # state -- h: (batch_size, num_layers, hidden_size)
        #          c: (batch_size, num_layers, hidden_size)
        i = torch.cat([x, context], dim=1)
        print('Input lstm cell: {}'.format(i.shape))
        h, c = self.lstm_cell(i, state)
        hidden = h
        state = (h, c)
        print('Output lstm cell --> hidden: {}, h, c: {}'.format(hidden.shape, state[0].shape))

        # ----------
        # Vocabulary
        # ----------
        # out -- (batch_size, vocab_size)
        out = self.linear(hidden)
        print('Out shape: {}'.format(out.shape))

        # -------
        # Softmax
        # -------
        # Shape after softmax:
        #    output -- (batch_size(1), vocab_size)
        out = F.log_softmax(out, dim=1)

        return out, state, alphas

    def forward(self, features, captions, lengths):
        """
        Forward decoder.

            :param features: float tensor of shape (batch_size, hidden_size==encoder_size)
            :param captions: long tensor of shape (batch_size, max_seq_length)
            :param lengths: long tensor of shape (batch_size, ) containing original length for each
                            caption.

        In training mode we use teacher forcing as we know the targets.
        """
        alphas_list = []
        predictions = []

        # ----------
        # Embeddings
        # ----------
        # captions -- (batch_size, max_length, embed_size)
        captions = self.embedding(captions)

        #
        # 
        #
        # hidden -- (batch_size, hidden_size)
        # fc1 = nn.Linear(encoder_size, hidden_size)ç
        # use F.linear instead!!!
        flat_features = features.view(features.size(0), -1)  # (batch_size, 7*7*encoder_size)
        fc1 = nn.Linear(flat_features.size(1), self.hidden_size).to('cuda')
        hidden = fc1(flat_features)
        #print('First hidden shape: {}'.format(hidden.shape))

        #
        # initial state
        #
        state = self.init_hidden(hidden)
        #print('State shape: h->{}, c->{}'.format(state[0].shape, state[1].shape))

        # -------------
        # Decoder steps
        # -------------
        # Using teacher forcing
        # FIXME: What happends withs PAD????
        for i in range(captions.size(1)):
            #print('===== IRETATION {} ====='.format(i + 1))
            embeddings = captions[:, i]
            #print('Embedding_size: {}'.format(embeddings.shape))

            # ---------
            # Attention
            # ---------
            # alphas -- (batch_size, num_pixels)
            # context -- (batch_size, encoder_size)
            context, alphas = self.attention(features, hidden)
            #print('Attention--> Context: {}, alphas: {}'.format(context.shape, alphas.shape))

            # ---------
            # LSTM step
            # ---------
            # hidden - (batch_size, hidden_size)
            # state -- h: (batch_size, num_layers, hidden_size)
            #          c: (batch_size, num_layers, hidden_size)
            i = torch.cat([embeddings, context], dim=1)
            #print('Input lstm cell: {}'.format(i.shape))
            h, c = self.lstm_cell(i, state)
            hidden = h
            state = (h, c)
            #print('Output lstm cell --> hidden: {}, h, c: {}'.format(hidden.shape, state[0].shape))

            # ----------
            # Vocabulary
            # ----------
            # out -- (batch_size, vocab_size)
            out = self.linear(hidden)
            #print('Out shape: {}'.format(out.shape))

            # Store predictions and alphas.
            predictions.append(out)
            alphas_list.append(alphas)

        # predictions -- shape (batch_size, max_length, vocab_size)
        # alphas -- shape (batch_size, max_length, num_pixels)
        predictions = torch.stack(predictions, dim=1)
        alphas_list = torch.stack(alphas_list, dim=1)
        return predictions, alphas_list