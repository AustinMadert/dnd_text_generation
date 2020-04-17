# -*- coding: utf-8 -*-
#!pip install tensorflow-gpu==2.0.0-beta1
import subprocess
import tensorflow as tf
import numpy as np
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Keras_Text_Generator(object):

    def __init__(self, batch_size=64, buffer_size=10000, checkpoint_dir='./training_checkpoints'):
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.model=tf.keras.Sequential()
        self.layers=[]
        self.dataset=None
        self.checkpoint_dir = checkpoint_dir
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")
        self.checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)


    def _load_data(self, filename):
        '''
        Opens and reads text file. Returns data as string.
        '''

        with open(filename, 'r') as f:
            data = f.read()

        return data


    def _vectorize_text(self):
        '''
        Creates indexes to transform characters to integers representation and back.
        Transforms text string into integer representation.
        '''

        # Create character to index (and vice versa) lookups
        self.char2idx = {j:i for i, j in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        # Transform text string into integer representation
        self.text_as_int = np.array([self.char2idx[char] for char in self.text_string])

        return None


    def _split_input_target(self, chunk):
        '''
        Takes a string and creates an input string and target string for training.
        '''

        input_text = chunk[:-1]
        target_text = chunk[1:]

        return input_text, target_text


    def _create_rolling_sequences(self):
        '''
        Takes text_string and creates seq_length sized training vectors by 
        incrementing through the text_string one character at a time.
        '''

        # Use pickling flag to determine whether to load from previously created
        # sequences
        if self.from_pickle:
            print('Opening pickled sequences...')
            with open(self.from_pickle, 'rb') as f:
                result = pickle.load(f)
            
            return result

        print('Generating new sequences...')

        dataX = []
        dataY = []

        # Iterate through full dataset
        for i in range(0, self.length_of_data - self.seq_length,
                        self.rolling_sequences_step):

            # Create segments of text that are seq_length long
            seq_in = self.text_string[i:i + self.seq_length]
            
            # Shift the segment forward one character
            seq_out = self.text_string[i + 1:i + self.seq_length + 1]

            # Add the inputs and outputs to lists as vector representations
            dataX.append([self.char2idx[char] for char in seq_in])
            dataY.append([self.char2idx[char] for char in seq_out])
        
        return (dataX, dataY)


    def _create_dataset(self):
        '''
        Checks for dataset creation method (as stored by the rolling_sequences
        variable). Then creates data segments and stores them as tensorflow
        Dataset objects. Finally it shuffles the data and organizes it into 
        training batches.        
        '''

        # Call rolling_sequences function and store in tensorflow Dataset object
        if self.rolling_sequences:
            self.data_raw = self._create_rolling_sequences()
            rolling_dataset = tf.data.Dataset.from_tensor_slices(self.data_raw)
            self.dataset = rolling_dataset.shuffle(self.buffer_size).batch(self.batch_size, 
                        drop_remainder=True)

        # If creating data chunks, then store vectorized data in tensorflow
        # Dataset object
        else:
            char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)

            # Break the Dataset into seq_length + 1 sized chunks of text
            sequences = char_dataset.batch(self.seq_length + 1, drop_remainder=True)

            # Map the chunking function across the entire dataset
            dataset = sequences.map(self._split_input_target)
            self.dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, 
                        drop_remainder=True)

        return None


    def load_and_create_dataset(self, filename, seq_length=100, 
            rolling_sequences=True, rolling_sequences_step=1, 
            from_pickle=None,
            return_raw=True):
        """Method designed to load a text file and create a training dataset. Data
        are loaded, then vectorized, and finally, the data is batched into training
        instances.
        
        Arguments:
            filename {str} -- full path and filename of data string.
        
        Keyword Arguments:
            seq_length {int} -- number of characters per training instance (default: {100})
            rolling_sequences {bool} -- flag used to determine data construction method (default: {True})
            rolling_sequences_step {int} -- if rolling_sequences, then the stride length to use (default: {1})
            from_pickle {bool} -- flag used to load previously created dataset (default: {False})
        
        Returns:
            None
        """

        # Load the data
        self.text_string = self._load_data(filename)

        # Store for later use
        self.seq_length = seq_length
        self.length_of_data = len(self.text_string)
        self.rolling_sequences = rolling_sequences
        self.from_pickle=from_pickle
        self.rolling_sequences_step=rolling_sequences_step

        # Create model character vocabulary
        self.vocab = sorted(set(self.text_string))
        self.vocab_size = len(self.vocab)

        print(f'Length of text: {self.length_of_data} characters')
        print(f'Unique characters: {self.vocab_size}')

        # Create char to index and index to char lookups, then text int representation
        self._vectorize_text()

        # Create the dataset using the rolling_sequences flag
        self._create_dataset()

        print('Dataset successfully created.')
        
        if return_raw:

            return self.data_raw

        return None


    def add_layer_to_model(self, layer, **kwargs):
        '''
        Takes layer type and number of units and adds new layer to model. Also 
        saves the layer and parameters in layers list as tuples for reassembly
        after training.
        '''

        # Store layers for use after training
        self.layers.append((layer, {**kwargs}))

        # Check if this is the first layer added: if so, set batch_size
        if len(self.model.layers) == 0:
            self.model.add(layer(batch_input_shape=[self.batch_size, None], **kwargs))
        else:
            self.model.add(layer(**kwargs))

        return None


    def _loss(self, labels, logits):
        '''
        Takes labels and logits from model and returns tf crossentropy loss function
        '''

        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


    def compile_model(self, optimizer='adam'):
        '''
        Take loss and optimizer inputs then compiles and displays model.
        '''

        self.model.compile(optimizer=optimizer, loss=self._loss)
        self.model.summary()

        return None

    
    def fit_model(self, epochs=10):
        '''
        Take number of epochs and fits model using class dataset. Saves best
        model using checkpoints.
        '''

        # Train the model
        history = self.model.fit(self.dataset, epochs=epochs,\
            callbacks=[self.checkpoint_callback])

        # Save model
        tf.train.latest_checkpoint(self.checkpoint_dir)

        return None


    def load_model_from_checkpoint(self):
        '''
        Loads model weights from existing checkpoint. Class model needs to have
        same architecture as the trained model.
        '''

        # Reset value for batch_input_shape to allow for starting text input
        self.layers[0][1]['batch_input_shape'] = [1, None]

        # Reassemble model used during training (this time with new batch shape)
        self.model = tf.keras.Sequential([
            layer(**kwargs) for layer, kwargs in self.layers
        ])

        # Load trained weights
        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))
        self.model.summary()

        return None


    def generate_text(self, start_string='<<start>>', temperature=1.0, num_generate=2000):
        """Once a model is trained or loaded, takes a start string and iteratively
        generates the specified number of output characters. Temperature controls
        the randomness of characters generated, or how suprising the characters
        are. Higher temperature means more surprising based on the dataset.
        
        Keyword Arguments:
            start_string {str} -- the initial inputs to the text generated. (default: {'<<start>>'})
            temperature {float} -- Controls how surprising generated text is. (default: {1.0})
            num_generate {int} -- Number of characters to generate. (default: {2000})
        
        Returns:
            Output Text [str] -- The generated text based on input.
        """

        # Vectorize start_string
        input_eval = [self.char2idx[char] for char in start_string]

        # Explicitly add first dimension to start_string making it a [1, 9] tensor
        input_eval = tf.expand_dims(input_eval, 0)
        
        text_generated = []
        
        self.model.reset_states()
        for _ in range(num_generate):

            # Get predictions from our model
            predictions = self.model(input_eval)
            # Reduce the dimensions of the predictions so that we can divide by 
            # temperature and create distribution
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions/temperature

            # Given our predictions distribution, randomly sample next char
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0]

            # Add a dimension to predicted char
            input_eval = tf.expand_dims([predicted_id], 0)

            # Turn char from vector representation to character representation 
            # and save
            text_generated.append(self.idx2char[predicted_id])
            
        # Join the start string with the predicted string
        self.output = (start_string + ''.join(text_generated))

        return self.output


    def create_start_string(self):
        start = '<<start>>'
        
        return None