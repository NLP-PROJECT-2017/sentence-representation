# sentence-representation

## introduction

This work presents two different models for encoding sentences, including DiscSent model and FastSent model. We evaluate sentence embeddings using SentEval on multiple tasks.


## proposal feedback

SB: This is a reasonable project topic, but I expect that you'll have speed issues with SkipThought. The standard version of the model, trained on Toronto Books, takes about two weeks to converge with one okay GPU. So, unless you have access to multiple good GPUs, you likely won't have time for any trial and error. I'd advise against just training both models for a shorter time: It's more important to know which one produces better representations than to know which one is faster to train. In real applications, you usually only need to train these once, so it's fine if it takes a long time.

To speed these up, consider either using much smaller representations for both models (100–300D) or switching to a smaller and more restricted source of text for training. Both of these will limit what we can learn, but you may have no better option. Feel free to stop by if you have any questions. You might also consider replacing SkipThought with a faster model, like FastSent from Hill et al.: https://github.com/fh295/SentenceRepresentation

KC: i agree with SB's concern on the computational overhead especially with skip-thought. some simplification should be sought after.
