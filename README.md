# product-separator

Serverless stem separation for Product. The worker uses the same two models
as the studio: BS-Roformer-SW, then MDX23C for the drum pieces.

RunPod builds this Dockerfile on its own machines. The weights are not in
git. While the image is built, RunPod downloads the public copies of
BS-Roformer-SW and the MDX23C drum model.

The endpoint stays at zero workers. A job starts a card, the card stops when
the job is done.

Input:

```json
{ "input": { "audio_url": "https://example.com/song.flac" } }
```

The endpoint needs `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`. Stems are
written to the `product-songs` bucket under `separations/<job id>/`.
