Real recordings, mixed into synthetic training by train_with_real_samples.py.

Organised per wake word: <model_name> below must match the model_name field
in that wake word's config/<model_name>.yaml.

  real_samples/<model_name>/train/
      Real recordings of the wake phrase, mixed into the synthetic positive
      training set. Copy your own WAV files here (one phrase per file -- use
      the WAV splitter tools in this repo's root if you recorded them
      back-to-back in one take).

  real_samples/<model_name>/eval_holdout/
      A few real recordings held back from training entirely, used to test
      the finished model automatically at the end of
      train_with_real_samples.py. Different takes to the ones in train/.

Both subfolders are created automatically the first time you run
train_with_real_samples.py and pick that wake word's config.
