# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is using  RandomForest the Classifier.

## Intended Use

The model determines whether an individual makes above or below $50,000 per year.

## Training Data

The data contains the following columns based: marital status, occupation, sex, salary, and other factors.

## Evaluation Data

The training data is sliced down to a smaller table using the same features, roughly 1/4 of the size of the original.

## Metrics

Precision: 0.7722 | Recall: 0.6036 | F1: 0.6775

## Ethical Considerations

If used to determine something such as eligibility for loans or mortgages, could discriminate against individuals based on race, education and other factors.

## Caveats and Recommendations

I would recommend using a newer dataset a lot has changed recently. Also, while RandomForestClassifier can be accurate, it can suffer from overfitting, which can influence new data.
