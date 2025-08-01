# HR Q&A Bot - Direct Data Approach

A Streamlit-based application that answers HR-related questions directly from data without using a fine-tuned model. This implementation uses semantic search with embeddings and fuzzy string matching to find the most relevant answers from a database of HR Q&A pairs.

## Features

- **Semantic Search**: Uses sentence embeddings to understand the meaning of questions
- **Fuzzy String Matching**: Handles typos and different phrasings of the same question
- **Hybrid Ranking**: Combines both methods for better results
- **No Model Training**: Works immediately with your Q&A data
- **Simple Interface**: Clean Streamlit UI with chat history

## How It Works

This approach works by:

1. **Loading Data**: HR questions and answers are loaded from a CSV file
2. **Generating Embeddings**: Each question is converted into a vector embedding using a pre-trained Sentence Transformer model
3. **Finding Matches**: When a user asks a question, the system:
   - Computes the semantic similarity between the user's question and all questions in the database
   - Performs fuzzy string matching to handle variations in wording
   - Combines both scores to find the best match
4. **Returning Answers**: The answer corresponding to the best-matched question is returned

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Ananaya-3396/HR-QnA-Bot.git
   cd HR-QnA-Bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements_direct_data.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run hr_direct_data_bot.py
   ```

## Project Structure

- `hr_direct_data_bot.py`: Main Streamlit application for the direct data approach
- `HR questions.csv`: Database of HR questions and answers
- `requirements_direct_data.txt`: Required Python packages for this approach

## Advantages of This Approach

- **No Training Required**: Works immediately with new data
- **Transparent**: You can see exactly which question matched
- **Lightweight**: Requires less computational resources than fine-tuned models
- **Deterministic**: Gives consistent answers
- **Easy to Update**: Just modify the CSV file to update or add new Q&A pairs

## Customization

### Adding New Q&A Pairs

To add new question-answer pairs, simply update the `HR questions.csv` file with new rows. The format should have two columns:
- `input`: The question
- `output`: The corresponding answer

### Adjusting Match Threshold

You can adjust the matching threshold in the `get_best_match` function to control how strict the matching should be. A higher threshold means only very close matches will be returned.

### Using Different Embedding Models

You can experiment with different Sentence Transformer models by changing the model name in the `load_embedding_model` function. Larger models may provide better semantic understanding but will be slower.

## License

This project is licensed under the MIT License - see the LICENSE file for details.