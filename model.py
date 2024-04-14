import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (вЉѓпЅЎвЂўМЃвЂївЂўМЂпЅЎ)вЉѓв”Ѓвњївњївњївњївњївњї
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.rnn = rnn_type(embed_size, hidden_size, rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        
        """
        YOUR CODE HERE (вЉѓпЅЎвЂўМЃвЂївЂўМЂпЅЎ)вЉѓв”Ѓвњївњївњївњївњївњї
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        embeddings = self.embedding(indices)
        packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False, batch_first=True)
        rnn_out, _ = self.rnn(packed)
        unpacked, _ = pad_packed_sequence(rnn_out, batch_first=True)
        logits = self.linear(unpacked)
        return logits


    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        """
        YOUR CODE HERE (вЉѓпЅЎвЂўМЃвЂївЂўМЂпЅЎ)вЉѓв”Ѓвњївњївњївњївњївњї
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        device = next(self.parameters()).device
        
        tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        tokens = torch.tensor(tokens, device=device).unsqueeze(0)

        if isinstance(self.rnn, nn.LSTM):
            hidden = (torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device),
                      torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device))
        else:
            hidden = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)
            
        while tokens.shape[1] < self.max_length:
            embeds = self.embedding(tokens[:, -1:])
            output, hidden = self.rnn(embeds, hidden)
            logits = self.linear(output) / temp
            new_token = Categorical(logits=logits[:, -1:]).sample()
            tokens = torch.cat([tokens, new_token], dim=1)
            
            if (new_token.item() == self.dataset.eos_id):
                break

        text = self.dataset.ids2text(tokens)[0]
        return text
