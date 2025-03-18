For the news sentiment analysis model, our team implemented a Bidirectional Long Short-Term Memory (BiLSTM). Unlike a standard LSTM, a BiLSTM is able to capture context from both directions of the text. In the context of news headlines, this is crucial because sentiment-modifying words can appear anywhere. A CNN would be an inappropriate choice for this task as headlines are inherently sequential in nature. A transformer model may have performed better, but it would have been a lot more complex and expensive to implement.

The model was trained on a dataset of just over 13,000 headlines from the GDELT dataset. As described in section \ref{news_data}, the labels were chosen from one of two datasets chosen as a hyperparameter.

The model is simple: it consists of 1 or 2 Bidrectional LSTM layers (determined by a hyperparameter) followed by a dropout layer and a fully connected layer.

The learning rate, weight decay, batch size, dropout rout, number of hidden dimensions, and number of BiLSTM layers are all hyperparameters. The model is being trained with Adam optimizer and the loss function of mean squared error.

Embedding is done using Word2Vec. The Word2Vec model is trained on the entire dataset with a vector size of 20 and a window size of 4 over 16 epochs.

\customref{fig:senti_curve}{Figure 5} below is the training and validation loss curve for the model using trained with labels from Gemini (\ref{news_data}) with hyperparameters of learning rate 0.0005, weight decay 0.0001, batch size 16, dropout 0.3, hidden dimension 64, and 2 BiLSTM layers.

With the same hyperparameters, the model infers the headline "Shareholders sell off 199,000 shares of Tesla, Tesla sued for $1.3 billion" to have a sentiment score of 0, and "Tesla expected to beat earnings, investors bullish" to have a sentiment score of 1.





The Global Database of Events, Language, and Tone (GDELT) Project analyzes global news media in many languages, storing headlines with associated "tone" scores, among other data. Its public API provided training data for the News Sentiment Analysis Model (\ref{news_model}), with headlines as inputs and tone scores as labels. Data was collected from GDELT so that each point of data used to point the model included the headline (the input) and the tone score (the label).

The API's \verb|artlist| mode has limitations: 250 results maximum per request. This means that the granularity of the news data would suffer for large date ranges. As well, the way in which the data is sorted is consequential. If asking the API to sort by ascending date, for example, looking for results across all of 2024 would only return the first 250 articles in 2024. To mitigate these factors, we queried monthly, sorting by relevance and using company names as search terms. We removed duplicate headlines through string matching with a 60\% similarity threshold. It was observed that roughly 15\% of the 250 returned results per request were deleted for being duplicates.

We collected labels using two methods, effectively forming two different datasets. First, GDELT tone values were obtained using the \verb|tonechart| mode since \verb|artlist| does not return tone data directly, despite being able to sort by it. However, the \verb|tonechart| mode does not return the tone of each article; it instead returns a histogram describing the distribution of tones for a given time frame. Articles are therefore grouped into tone "buckets" (e.g. 15 articles in a specified time frame belong to the tone bucket with value -3). There is no way to specify the number of articles returned by the API, and not all articles are returned. After cleaning duplicates, headlines were cross-referenced with tone data. Almost 50\% of unique headlines were discarded due to missing tone values. Remaining tone scores were clamped to $[-20, 20]$ and then linearly mapped to $[-1, 1]$.

A limitation of GDELT tone scores is they are based on full article content, not just headlines. Therefore, we implemented a second labeling method using Google's Gemini LLM. In batches of 120, headlines were programmatically scored for sentiment in the range $[-1, 1]$. This approach based labels solely on headlines and retained 100\% of unique headlines. Qualitatively, we observed that the Gemini labels more effectively captured sentiment in headlines than the GDELT tone labels.

Both datasets were maintained, with the choice between them treated as a hyperparameter during model training (\ref{news_model}). The Gemini-labeled dataset performed significantly better.