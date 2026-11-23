# product-separator

Serverless stem separation for Product. The worker uses the same two models
as the studio: BS-Roformer-SW, then MDX23C for the drum pieces.

RunPod builds this Dockerfile. The weights are not in git. They are the files
already used by the studio, attached to the `models` release, and the image
downloads them while it is built.

The endpoint stays at zero workers. A job starts a card, the card stops when
the job is done.

Input:

```json
{ "input": { "audio_url": "https://example.com/song.flac" } }
```

The endpoint needs `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`. Stems are
written to the `product-songs` bucket under `separations/<job id>/`.
