import jax
import hm_attention
import pytest


@pytest.mark.parametrize("dim", [8, 32, 64])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize('seq_length', [10, 16, 45])
@pytest.mark.parametrize('batch_size', [1, 4, 16, 32])
def test_attention(dim, heads, seq_length, batch_size):
    random_key = jax.random.PRNGKey(seed=42)
    object_creation_key, random_key = jax.random.split(random_key)
    attention_block = hm_attention.SelfAttention.factory(
        object_creation_key, dim_io=dim, dim_ff=dim, num_heads=heads)
    input = jax.random.normal(
        key=random_key, shape=(batch_size, seq_length, dim))
    output = attention_block.forward(input)
    assert output.shape == (batch_size, seq_length, dim)
    # print('------------------------')
    # print(input)
    # print('sssssssssssssssssssssssss')
    # print(output)
    # print('xxxxxxxxxxxxxxxxxxxxxxxxx')
    # assert False


@pytest.mark.parametrize("dim", [8, 32, 64])
@pytest.mark.parametrize('seq_length', [10, 16, 45])
@pytest.mark.parametrize('batch_size', [1, 4, 16, 32])
def test_layer_norm(dim, seq_length, batch_size):
    random_key = jax.random.PRNGKey(seed=42)
    norm = hm_attention.LayerNorm.factory(max_tokens=seq_length)
    input = jax.random.normal(
        key=random_key, shape=(batch_size, seq_length, dim))
    output = norm.forward(input)
    assert output.shape == (batch_size, seq_length, dim)
