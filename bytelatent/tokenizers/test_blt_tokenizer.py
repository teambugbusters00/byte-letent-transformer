# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import pytest

from bytelatent.constants import BLT_DATA
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.tokenizers.word_level_tokenizer import WordLevelTokenizer
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs


def test_tokenizer_bytes():
    with open("fixtures/tokenizer_data.json") as f:
        data = json.load(f)

    examples: list[str] = data["texts"]
    examples_tokens: list[list[int]] = data["tokens"]

    tokenizer = BltTokenizer(bpe_delim=False)
    for i in range(len(examples)):
        assert tokenizer.encode(examples[i]) == examples_tokens[i]


def test_tokenizer_bpe():
    with open("fixtures/tokenizer_data_bpe_delim.json") as f:
        data = json.load(f)

    examples: list[str] = data["texts"]
    examples_tokens: list[list[int]] = data["tokens"]

    tokenizer = BltTokenizer(bpe_delim=True)
    for i in range(len(examples)):
        assert tokenizer.encode(examples[i]) == examples_tokens[i]


def test_build_tokenizer_from_args():
    tokenizer_args = TokenizerArgs(
        name="blt",
        init_kwargs={
            "bpe_tokenizer_path": BLT_DATA / "tokenizer_final_32k.minus_inf_ws.model"
        },
    )
    tokenizer = tokenizer_args.build()
    assert tokenizer.encode("test text") is not None


class TestWordLevelTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return WordLevelTokenizer(lowercase=False)
    
    @pytest.fixture
    def lowercase_tokenizer(self):
        return WordLevelTokenizer(lowercase=True)
    
    def test_basic_tokenization(self, tokenizer):
        text = "Hello, world! This is a test."
        # The tokenizer now preserves whitespace, so we expect more tokens
        expected_tokens = ["Hello", ",", " ", "world", "!", " ", "This", " ", "is", " ", "a", " ", "test", "."]
        
        # Test encoding
        token_ids = tokenizer.encode(text)
        assert len(token_ids) == len(expected_tokens)
        
        # Test decoding
        decoded = tokenizer.decode(token_ids)
        assert decoded == text  # Should match original text exactly
        
        # Test roundtrip
        assert tokenizer.decode(tokenizer.encode(text)) == text
    
    def test_lowercase_tokenization(self, lowercase_tokenizer):
        text = "Hello, World!"
        token_ids = lowercase_tokenizer.encode(text)
        decoded = lowercase_tokenizer.decode(token_ids)
        assert decoded == "hello, world!"
    
    def test_special_tokens(self, tokenizer):
        # Test BOS/EOS
        text = "test"
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        assert len(token_ids) == 3  # BOS + token + EOS
        assert token_ids[0] == tokenizer.vocab["<bos>"]
        assert token_ids[-1] == tokenizer.vocab["<eos>"]
        assert token_ids[1] == tokenizer.vocab["test"]
    
    def test_punctuation_handling(self, tokenizer):
        text = "Hello, world! How's it going?"
        token_ids = tokenizer.encode(text)
        assert "," in tokenizer.inverse_vocab.values()
        assert "!" in tokenizer.inverse_vocab.values()
        assert "'" in tokenizer.inverse_vocab.values()
        assert "?" in tokenizer.inverse_vocab.values()
    
    def test_vocab_growth(self, tokenizer):
        initial_vocab_size = tokenizer.get_vocab_size()
        
        # Add a new word
        text = "supercalifragilisticexpialidocious"
        token_ids = tokenizer.encode(text)
        
        # Check vocab grew by 1
        assert tokenizer.get_vocab_size() == initial_vocab_size + 1
        
        # Check we can decode it back
        assert tokenizer.decode(token_ids) == text
    
    def test_token_offsets(self, tokenizer):
        text = "Hello, world!"
        tokens, offsets = tokenizer.get_token_offsets(text)
        
        # With whitespace preservation, we expect 5 tokens: "Hello", ",", " ", "world", "!"
        assert len(tokens) == 5
        assert len(offsets) == 5
        
        # Check that offsets are in order
        assert offsets == sorted(offsets)
        
        # Check that tokens can be reconstructed to original text
        reconstructed = "".join(tokens)
        assert reconstructed == text
