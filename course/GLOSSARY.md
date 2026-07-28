# Course Glossary

- **Active latent dimension:** A VAE dimension whose average raw KL exceeds the
  diagnostic threshold, suggesting it carries input-dependent information.
- **Aggregate posterior:** The distribution of encoded $z$ values after
  averaging $q(z|x)$ over the data distribution.
- **Bottleneck:** A constraint that limits what information can pass from input
  to reconstruction.
- **Codebook:** The learned set of embedding vectors available to a VQ-VAE.
- **Codebook collapse:** A failure in which only a small subset of available
  codes receives assignments.
- **Commitment loss:** VQ loss encouraging encoder outputs to remain near their
  selected codebook embeddings.
- **Decoder:** Maps a latent representation back to the observation space.
- **Distortion:** Reconstruction cost in rate–distortion language.
- **ELBO:** Evidence lower bound; the VAE objective whose negative combines
  reconstruction and KL.
- **Free bits:** A per-dimension KL allowance below which the objective supplies
  no additional compression gradient.
- **Latent:** An internal representation used to reconstruct or generate data.
- **Posterior collapse:** A VAE failure where $q(z|x)$ approaches the prior
  and the decoder ignores $z$.
- **Prior:** A distribution over latent variables before observing an input.
- **Rate:** VAE KL in nats; an information-cost interpretation of the latent.
- **Reparameterization trick:** Expresses stochastic VAE samples as a
  differentiable transformation of parameters and parameter-free noise.
- **Straight-through estimator:** Uses a discrete/quantized value forward while
  approximating its backward gradient as identity.
- **Vector quantization:** Replaces a continuous vector with the nearest learned
  codebook embedding.

