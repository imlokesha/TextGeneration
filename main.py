import tensorflow as tf
import numpy as np
import requests
import time
import os
import keras
# import tensorflow.keras as k2

"""GPU and CPU Check"""
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("Default GPU Device: ", tf.test.gpu_device_name())


# print("CPU LIST:", tf.config.list_physical_devices("CPU"))
# print("GPU LIST:", tf.config.list_physical_devices("GPU"))
# print("BUILD WITH CUDA:", tf.test.is_built_with_cuda())  # Installed non gpu package
# print("Deprecated AVAILABLE:", tf.test.is_gpu_available())  # Deprecated
# print("Deprecated AVAILABLE:", tf.test.is_gpu_available(cuda_only=False))  # Deprecated


# print(tf.__version__)


def download_shakespeare_text():
    """
    This function downloads the Shakespeare text from a given URL and saves it to a local file.

    Parameters:
    None

    Returns:
    None

    The function performs the following steps:
    1. Sets the current working directory as the root directory.
    2. Creates a "data" directory if it does not exist.
    3. Defines the URL from which to download the Shakespeare text.
    4. Defines the local file path where the downloaded text will be saved.
    5. Sends a GET request to the URL to retrieve the text.
    6. Writes the retrieved text to the local file.
    """
    ROOT = os.getcwd()
    os.chdir(ROOT)
    os.makedirs("data", exist_ok=True)

    url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
    file_path = "./data/shakespeare.txt"

    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)



"""samething can be done using config.yaml"""
class Config:
  path_to_file = os.path.join("data", "shakespeare.txt")
  seq_length = 100

  Batch_size = 64
  Buffer_size = 10000 # 

  embedding_dim = 256

  rnn_units = 1024

  EPOCHS = 30

  checkpoint_dir = "./training_ckpt"

Config.EPOCHS



text = open(Config.path_to_file, "rb").read().decode(encoding='utf-8')
text[:100] 
"""above or below code can be used""" 
# with open(Config.path_to_file) as f:
#   t = f.read()

# t[:100]



"""*Finding out unique charaters"""
set(text) # we get all the letters
vocab = sorted(set(text)) #dones on alphabhetical order
len(vocab)


"""Index and character
Passing character and getting index for that
"""

char2idx = {uniChar: idx for idx, uniChar in enumerate(vocab)}
char2idx



"""
We cannot pass txt values
We need to convert text to integer
"""
text_as_int = np.array([char2idx[c] for c in text])
text_as_int # all text is now represented as integer


text[:13], text_as_int[:13]
len(text)

"""Total length Divided by seq length"""
examples_per_epoch = len(text)//(Config.seq_length + 1)
examples_per_epoch

idx2char_DICT = {val: key for key, val in char2idx.items()}

"""Iterating character to index"""
idx2char = np.array(vocab)
idx2char

idx2char[0], idx2char_DICT[0] 


"""Creating dataset for training part

Passing an entire array to create as a dataset"""
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for i in char_dataset.take(13):
  print(idx2char[i.numpy()])



"""drop_remainder - if there is extra sequence, it'll drop

Char dataset converted to batch of 101"""

sequences = char_dataset.batch(Config.seq_length + 1, drop_remainder=True)

for item in sequences.take(10):
  to_print = repr("".join([idx2char[c] for c in item.numpy()]))
  print(to_print)
  print(len(to_print))



"""# First Citize -> irst Citizen
Making target(output) and input valiue"""
def split_input_target(chunk):
    """
    This function splits a given chunk of text into input and target sequences for training a text generation model.
    The input sequence consists of all characters except the last one, while the target sequence consists of all characters except the first one.

    Parameters:
    chunk (tf.Tensor): A tensor representing a chunk of text. The tensor should be a 1D array of integers, where each integer represents a character.

    Returns:
    tuple: A tuple containing two tensors: input_text and target_text.
        - input_text (tf.Tensor): A tensor representing the input sequence. It contains all characters except the last one.
        - target_text (tf.Tensor): A tensor representing the target sequence. It contains all characters except the first one.
    """
    input_text = chunk[:-1] # First Citize
    target_text = chunk[1:] # irst Citizen
    return input_text, target_text

dataset = sequences.map(split_input_target)




for input_example, target_example in dataset.take(1):
  print("input_data:\n")
  print(repr("".join([idx2char[i] for i in input_example.numpy()])))
  print("\n\ntarget_data:\n")
  print(repr("".join([idx2char[t] for t in target_example.numpy()])))


"""Creating training batches"""
dataset = dataset.shuffle(Config.Buffer_size).batch(Config.Batch_size, drop_remainder=True)
dataset


Config.vocab_size = len(vocab)
Config.vocab_size



def build_model(
    vocab_size, embedding_dim, rnn_units, batch_size):
    """
    This function constructs a recurrent neural network (RNN) model for text generation.
    The model consists of an embedding layer, a GRU layer, and a dense layer.

    Parameters:
    vocab_size (int): The size of the vocabulary, i.e., the number of unique characters in the text.
    embedding_dim (int): The dimension of the embedding layer, representing the size of the dense vector used to represent each character.
    rnn_units (int): The number of units in the GRU layer, determining the complexity of the model's memory.
    batch_size (int): The size of the input batches, representing the number of samples processed at once during training.

    Returns:
    tf.keras.models.Model: The constructed RNN model for text generation.
    """
    model = tf.keras.Sequential([
          tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
          tf.keras.layers.GRU(rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'),
          tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size = Config.vocab_size,
    embedding_dim = Config.embedding_dim,
    rnn_units = Config.rnn_units,
    batch_size = Config.Batch_size
)



# print(model.summary())


"""To train the model, we have to define the loss"""
def loss(labels, logits):
    """
    Calculates the cross-entropy loss between the true labels and predicted logits.

    This function is used as the loss function for training a neural network model in TensorFlow.
    It computes the cross-entropy loss between the true labels and the predicted logits,
    which are the output values of the model before applying the softmax function.

    Parameters:
    labels (tf.Tensor): A tensor containing the true labels for the input data.
        The shape of the tensor should be [batch_size, sequence_length], where
        batch_size is the number of samples in the batch and sequence_length is the length of the sequence.
        The values in the tensor should be integers representing the true labels.
    logits (tf.Tensor): A tensor containing the predicted logits for the input data.
        The shape of the tensor should be [batch_size, sequence_length, num_classes], where
        batch_size is the number of samples in the batch, sequence_length is the length of the sequence,
        and num_classes is the number of possible classes for the labels.

    Returns:
    tf.Tensor: A scalar tensor representing the cross-entropy loss between the true labels and predicted logits.
    """
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)



checkpoint_prefix = os.path.join(Config.checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    save_weights_only = True
)


"""For every EPOCH it creates checkpoint"""
history = model.fit(dataset, epochs=Config.EPOCHS, callbacks=[checkpoint_callback])

"""Steps per EPOCH"""
(len(text)/Config.Batch_size)/(Config.seq_length + 1)
                              

# Restoring Latest checkoint -
tf.train.latest_checkpoint(Config.checkpoint_dir) #to get latest checkpoint


"""We'll load the known weights into the model"""
model_from_ckpt = build_model(
    vocab_size = Config.vocab_size,
    embedding_dim = Config.embedding_dim,
    rnn_units = Config.rnn_units,
    batch_size = 1 # during prediction we don't pass batch of data, Only one
)


# print(model_from_ckpt.summary())


model_from_ckpt.load_weights(tf.train.latest_checkpoint(Config.checkpoint_dir))

model_from_ckpt.build(tf.TensorShape([1, None]))


print(model_from_ckpt.summary())




"""Pediction"""
# define a 
def generate_text(model, start_string, no_of_chars_to_gen=1000):
    """
    This function generates text based on a trained model and a starting string.
    It uses the model to predict the next character in the sequence and appends it to the generated text.
    The process is repeated for a specified number of characters.

    Parameters:
    model (tf.keras.models.Model): The trained model used for text generation.
    start_string (str): The initial string from which the text generation starts.
    no_of_chars_to_gen (int, optional): The number of characters to generate. Defaults to 1000.

    Returns:
    str: The generated text, starting with the provided start_string and followed by the generated characters.
    """
    # convert the input text to nos.
    input_val = [char2idx[s] for s in start_string] # text converted to int
    input_val = tf.expand_dims(input_val, 0) # [] ->> [1, ]

    text_generated = list()

    temperature = 1.0 #more predicatable or surprising text

    # Resetting the previous states if any while predictions.
    model.reset_states() 

    for i in range(no_of_chars_to_gen):
        predictions = model(input_val)

        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        # print(predictions)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # print(predicted_id)
        
        input_val = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + "".join(text_generated)



result = generate_text(model=model_from_ckpt, start_string="SUNNY")
print(result)


tf.math.log([[0.5, 0.5]])



for _ in range(15):
  predictions = [[1000., 1.,2.,3.,4., 55., 56., 100., 101., 200., 1001]]
  samples = tf.random.categorical(predictions, 1)[-1, 0]
  print(samples.numpy())
  # print(values[samples.numpy()])