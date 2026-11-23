# product-separator

Serverless stem separation for Product. The worker uses the same two models
as the studio: BS-Roformer-SW, then MDX23C for the drum pieces.

RunPod builds this Dockerfile on its own machines. The weights are not in
git. While the image is built, RunPod downloads the public copies of
BS-Roformer-SW and the MDX23C drum model.

The endpoint stays at zero workers. A job starts a card, the card stops when
the job is done.

Input for a separation:

```json
{ "input": { "audio_url": "https://example.com/song.flac", "clip_id": "the studio clip id" } }
```

`clip_id` is the studio clip. The worker writes one box per drum hit and one box per note to `notes/<clip_id>.json` in the same bucket. A separation without that file is not finished.

Input for lyrics. The Hebrew Whisper model hears the vocal on the same card
and returns each line with its start and end, in seconds:

```json
{ "input": { "task": "lyrics", "audio_url": "https://example.com/vocals.flac" } }
```

The endpoint needs `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`. Stems are
written to the `product-songs` bucket under `separations/<job id>/`.
