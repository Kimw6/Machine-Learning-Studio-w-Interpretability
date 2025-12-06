import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from local_components import card_container
import time
import utilityFunctions as progress

class NeuralNetworkModel:
    def __init__(self):
        self.params = {
            'num_layers': 2,
            'random_state': 42,
            'epochs': 10,
        }
        self.model = None
        self.num_classes = None
        self.loss_fn = None
        self.y_train_raw = None
        self.y_test_raw = None

        # make sure flags exist
        if 'trained_nn' not in st.session_state:
            st.session_state['trained_nn'] = False
        if 'models' not in st.session_state:
            st.session_state['models'] = {}
        if 'Neural Network' not in st.session_state['models']:
            st.session_state['models']['Neural Network'] = None

    def parameters(self):
        """Show the NN config form when not trained."""
        if not st.session_state.get('trained_nn', False):
            st.write("Please configure your neural network below.")
            num_classes = len(st.session_state['df'][st.session_state['target']].unique())

            with st.form(key='nn_model_form'):
                st.write(f"Current dataset has `{num_classes}` classes for `{st.session_state['target']}`.")
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Number of Hidden Layers", min_value=1, max_value=5, value=2, key='num_layers')
                    st.number_input("Epochs", min_value=1, max_value=100, value=10, key='epochs')
                with col2:
                    st.number_input("Random State", min_value=1, max_value=1000, value=42, key='random_state')

                st.form_submit_button("Train Model", on_click=self.on_click)

    def on_click(self):
        progress.set_done("qs_train", True)
        progress.set_done("qs_interpret", True)

        # collect params from session
        for k in self.params.keys():
            self.params[k] = st.session_state[k]

        try:
            self.build_model()
            self.train_model()
            st.session_state['trained_nn'] = True
        except Exception as e:
            st.error(f"Training failed: {e}")

    def build_model(self):
        df = st.session_state['df']
        target = st.session_state['target']

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(target, axis=1), df[target],
            test_size=0.2, random_state=self.params['random_state']
        )

        self.y_train_raw = y_train
        self.y_test_raw = y_test
        self.num_classes = len(np.unique(y_train))

        model = Sequential()
        total_units = max(128, 16 * self.params['num_layers'])
        model.add(Dense(units=total_units, activation='relu', input_dim=X_train.shape[1]))

        for _ in range(self.params['num_layers']):
            total_units = max(16, total_units // 2)
            model.add(Dense(units=total_units, activation='relu'))

        if self.num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            self.loss_fn = 'binary_crossentropy'
        else:
            model.add(Dense(self.num_classes, activation='softmax'))
            self.loss_fn = 'categorical_crossentropy'

        model.compile(optimizer='adam', loss=self.loss_fn, metrics=['accuracy'])
        self.model = model

        # short spinner for UX
        with st.spinner(text="Building Neural Network...", show_time=True):
            time.sleep(1.0)

    def train_model(self):
        df = st.session_state['df']
        target = st.session_state['target']

        X_train, X_test, _, _ = train_test_split(
            df.drop(target, axis=1), df[target],
            test_size=0.2, random_state=self.params['random_state']
        )

        # labels
        if self.num_classes == 2:
            y_train = self.y_train_raw.values.reshape(-1, 1)
            y_test = self.y_test_raw.values.reshape(-1, 1)
        else:
            y_train = tf.keras.utils.to_categorical(self.y_train_raw)
            y_test = tf.keras.utils.to_categorical(self.y_test_raw)

        history = self.model.fit(
            X_train, y_train,
            epochs=self.params['epochs'],
            validation_data=(X_test, y_test),
            verbose=0
        )

        self.save_model_results('Neural Network', history)

    def plot(self, history, metric_type='loss'):
        if metric_type == 'loss':
            train_vals = history.history['loss']
            val_vals = history.history['val_loss']
            title = 'Loss Over Epochs'
            y_label = 'Loss'
        else:
            train_vals = history.history['accuracy']
            val_vals = history.history['val_accuracy']
            title = 'Accuracy Over Epochs'
            y_label = 'Accuracy'

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_vals) + 1), train_vals, label='Train', marker='o')
        plt.plot(range(1, len(val_vals) + 1), val_vals, label='Validation', marker='o')
        plt.title(f"Training and Validation {title}")
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    def save_model_results(self, model_name, history):
        st.session_state['models'][model_name] = {
            'model': self.model,
            'name_of_model': model_name,
            'num_classes': self.num_classes,
            'y_train': self.y_train_raw,
            'y_test': self.y_test_raw,
            'trained': True,
            'history': history,
        }

    def display_results(self, model_data):
        with st.expander("View Neural Network Training Results", expanded=True):
            st.subheader("Model Configuration & Architecture")
            with card_container(key="nn_results_info"):
                st.write("")
                st.markdown(f"**Model Name**: {model_data['name_of_model']}")
                st.markdown(f"**Target Classes**: {model_data['num_classes']}")
                model_data['model'].summary(print_fn=lambda x: st.text(x))

        with st.expander("View Neural Network Loss & Accuracy Plots", expanded=True):
            st.subheader("Training & Validation Loss")
            with card_container(key="loss_plots"):
                st.write("")
                self.plot(model_data['history'], 'loss')

            st.subheader("Training & Validation Accuracy")
            with card_container(key="accuracy_plots"):
                st.write("")
                self.plot(model_data['history'], 'accuracy')

    def reset_model(self):
        """Reset ONLY the Neural Network model and its flags, then rerun."""
        st.session_state['models']['Neural Network'] = None
        st.session_state['trained_nn'] = False
        st.rerun()
