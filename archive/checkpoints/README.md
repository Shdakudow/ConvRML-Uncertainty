# Published checkpoint catalog

`checkpoint_index.json` lists release URLs, sizes and SHA-256 hashes for published
weights. It contains no model parameters and is not needed for training or for
inference with an existing checkpoint path.

The optional `python -m convrml.download` command reads this catalog from the
repository checkout. Keep it here if you use that command. The actual `.pt`
weights remain in their existing releases; archiving the catalog does not move
or delete those weights.
