# HF Spaces Deploy Procedure

The three files in this directory (`Dockerfile`, `entrypoint.sh`, `README.md`)
are the source-of-truth for the Hugging Face Space. They get pushed to a
separate HF Space repo (not to docagent). The Dockerfile clones the docagent
repo from GitHub at build time, so the Space repo itself stays tiny.

## One-time setup

1. **Create the Space**
   - Go to https://huggingface.co/new-space
   - Owner: your HF account
   - Space name: `docagent` (or whatever)
   - License: MIT (matches docagent)
   - SDK: **Docker** → "Blank" template
   - Hardware: CPU basic (free)
   - Visibility: Public

2. **Clone the empty Space repo locally**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/docagent hf-docagent
   cd hf-docagent
   ```

3. **Copy the three files in**
   ```bash
   cp /c/Users/nachung/projects/docagent/hf-space/Dockerfile .
   cp /c/Users/nachung/projects/docagent/hf-space/entrypoint.sh .
   cp /c/Users/nachung/projects/docagent/hf-space/README.md .
   ```

4. **Set the secret** (Space → Settings → Variables and secrets)
   - Name: `ANTHROPIC_API_KEY`
   - Value: your Anthropic API key
   - Mark as secret (not variable)

5. **Push to trigger the build**
   ```bash
   git add Dockerfile entrypoint.sh README.md
   git commit -m "Initial deploy"
   git push
   ```

6. **Watch the build** on the Space's "Logs" tab. Expect ~10 min:
   - Pull pgvector/pgvector:pg16 base (~300 MB)
   - Install Python 3.11 + apt deps
   - Clone docagent from GitHub
   - Install CPU torch + requirements (~700 MB)
   - Pre-download HF models (~1.2 GB)
   - Final image ~5 GB

7. **First request** takes ~60s while the corpus re-ingests on cold start.
   Subsequent requests <1s.

## Updating the deploy

The Dockerfile clones docagent's `main` branch. To pick up new changes:

1. Push the change to `github.com/NathanChung4/docagent`
2. Go to the Space → Settings → "Factory reboot" (forces image rebuild)

If you change one of the three deploy files (`Dockerfile`, `entrypoint.sh`,
`README.md`):

1. Update the file in `docagent/hf-space/` (source-of-truth)
2. Copy it into the HF Space repo
3. `git push` to the Space — rebuilds automatically

## Pinning to a specific commit

The Dockerfile takes a `DOCAGENT_REF` build arg (default `main`):

```dockerfile
ARG DOCAGENT_REF=main
RUN git clone --depth=1 --branch ${DOCAGENT_REF} ...
```

To pin: edit the Dockerfile and set `ARG DOCAGENT_REF=v1.2.3` (any tag,
branch, or commit SHA), then push.

## Troubleshooting

**Build fails: "postgres: command not found"**
- The `PATH` in the runtime stage prepends `/usr/lib/postgresql/16/bin`. If
  this changes in a future pgvector base image, update the Dockerfile.

**Build fails: "permission denied" on /tmp/pgdata**
- The container runs as UID 1000 (`user`). `/tmp` is world-writable by
  default; if HF Spaces' base image ever changes that, switch to a path
  under `/home/user/pgdata`.

**Streamlit shows "API unreachable"**
- The API takes 5-10s after container start to bind :8000. The dashboard
  caches the reachability check for 30s. Refresh the page after a minute.

**"I don't know" on every query**
- Confirm `ANTHROPIC_API_KEY` is set in Space secrets (not variables).
  Retrieval still works without it; generation doesn't.

**Cold start is slow**
- HF free-tier Spaces sleep after 48h idle. First request after sleep
  pays the ~60s ingest cost. To avoid: upgrade to a persistent Space, or
  hit the Space periodically (a cron job or HF's "keep-alive" feature).
