predicted = tf.round(tf.nn.sigmoid(logits))
actual = labels
    
# Count true positives, true negatives, false positives and false negatives.
tp = tf.count_nonzero(predicted * actual)
tn = tf.count_nonzero((predicted - 1) * (actual - 1))
fp = tf.count_nonzero(predicted * (actual - 1))
fn = tf.count_nonzero((predicted - 1) * actual)
    
# Calculate accuracy, precision, recall and F1 score.
accuracy = (tp + tn) / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
fmeasure = (2 * precision * recall) / (precision + recall)

# Add metrics to TensorBoard.    
tf.summary.scalar('Accuracy', accuracy)
tf.summary.scalar('Precision', precision)
tf.summary.scalar('Recall', recall)
tf.summary.scalar('f-measure', fmeasure)
