



import pandas as pd

from src.agent.gemini_agent import is_plasma


if __name__ == '__main__':
    df = pd.read_csv("../data/plasma_test_data_annotated_.csv", encoding="utf-8")
    print(df.head())
    df['gemini_label'] = df['text'].map(lambda line: is_plasma(line))
    df.to_csv("../data/plasma_test_data_with_gemini_label.csv", encoding='utf-8',
              index=False)


