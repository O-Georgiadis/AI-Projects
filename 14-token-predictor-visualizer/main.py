from visualizer import TokenPredictor, create_token_graph, visualize_predictions

OPENAI_MODEL = "gpt-4.1-mini"


message = "In one sentence, describe the color red to someone who has never been able to see"


predictor = TokenPredictor(OPENAI_MODEL)
predictions = predictor.predict_tokens(message)
G = create_token_graph(OPENAI_MODEL, predictions)
plt = visualize_predictions(G)
plt.show()