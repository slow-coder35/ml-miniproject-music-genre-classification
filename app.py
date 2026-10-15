"""Local music genre demonstration. Launch with scripts/run.sh app."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'src'))
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from config import RESULTS_DIR,SR,WINDOW_SAMPLES
from features import log_mel_window
from predict import Predictor,decode_upload

st.set_page_config(page_title='Music Genre Classifier',page_icon='🎵',layout='wide')
st.title('Music Genre Classifier')
st.write('Listen to a music clip, explore its sound, and predict its genre with a convolutional neural network.')
st.caption('Ten GTZAN genres · Local inference · Five short windows per recording')

@st.cache_resource
def get_predictor(selection_modified):
    return Predictor()

upload=st.file_uploader('Choose a WAV or FLAC recording',type=['wav','flac'],max_upload_size=20)
if upload:
    content=upload.getvalue()
    st.audio(content)
    try:
        audio,details=decode_upload(content)
        st.caption(f'Analyzing {details["analyzed_seconds"]:.1f} seconds of a {details["source_duration_seconds"]:.1f}-second recording.')
        left,right=st.columns(2)
        with left:
            fig,ax=plt.subplots(figsize=(7,2.6))
            step=max(1,len(audio)//12000)
            ax.plot(np.arange(0,len(audio),step)/SR,audio[::step],color='#007f8b',linewidth=.6)
            ax.set(xlabel='Time (seconds)',ylabel='Amplitude',title='Waveform');fig.tight_layout();st.pyplot(fig);plt.close(fig)
        with right:
            start=(len(audio)-WINDOW_SAMPLES)//2
            mel=log_mel_window(audio[start:start+WINDOW_SAMPLES])
            fig,ax=plt.subplots(figsize=(7,2.6))
            img=librosa.display.specshow(mel,sr=SR,hop_length=256,x_axis='time',y_axis='mel',ax=ax,cmap='magma')
            ax.set_title('Central two-second log-mel spectrogram');fig.colorbar(img,ax=ax,label='dB');fig.tight_layout();st.pyplot(fig);plt.close(fig)
        if st.button('Predict genre',type='primary'):
            selection=RESULTS_DIR/'demo_model.json'
            if not selection.exists(): st.error('The trained model bundle is missing. Complete training and model selection first.')
            else:
                with st.spinner('Analyzing the music…'):
                    result=get_predictor(selection.stat().st_mtime_ns).predict_audio(audio,details)
                st.subheader(result['genre'].replace('hiphop','hip-hop').title())
                st.caption(f'Model: {result["model"]} · {result["window_count"]} windows · Scores express model preference and may be overconfident.')
                table=pd.DataFrame(result['top_three']).rename(columns={'genre':'Genre','score':'Model score'})
                st.dataframe(table,hide_index=True,column_config={'Model score':st.column_config.ProgressColumn(min_value=0,max_value=1,format='percent')})
                st.bar_chart(pd.DataFrame({'Genre':list(result['scores']),'Model score':list(result['scores'].values())}).set_index('Genre'))
    except (ValueError,RuntimeError) as error:
        st.error(str(error))
else:
    st.info('Upload at least two seconds of music. The app analyzes up to the first 30 seconds.')

with st.expander('About the project'):
    st.write('The project compares SVM, k-nearest neighbors, a feed-forward neural network, and a CNN on raw audio and log-mel spectrograms. The displayed prediction uses the mel CNN selected using validation data.')
    st.write('Genres overlap, and a small historical dataset cannot represent all music. Speech, silence, mixed genres, and unfamiliar recordings may produce unreliable predictions.')
