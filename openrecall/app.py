from threading import Thread

import numpy as np
from pathlib import Path

from flask import Flask, abort, render_template_string, request, send_from_directory
from jinja2 import BaseLoader

from openrecall.config import appdata_folder, screenshots_path
from openrecall.database import create_db, get_all_entries, get_entry_by_id, get_timestamps
from openrecall.nlp import cosine_similarity, get_embedding
from openrecall.screenshot import record_screenshots_thread
from openrecall.utils import human_readable_time, timestamp_to_human_readable

app = Flask(__name__)

app.jinja_env.filters["human_readable_time"] = human_readable_time
app.jinja_env.filters["timestamp_to_human_readable"] = timestamp_to_human_readable


def _capture_filenames_for_timestamp(timestamp: int) -> list[str]:
    screenshots_dir = Path(screenshots_path)
    matches = sorted(screenshots_dir.glob(f"{int(timestamp)}*.webp"))
    if not matches:
        for suffix in (".png", ".jpg", ".jpeg"):
            matches.extend(sorted(screenshots_dir.glob(f"{int(timestamp)}*{suffix}")))
    return [path.name for path in sorted(matches)]

base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenRecall</title>
  <!-- Bootstrap CSS -->
  <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.3.0/font/bootstrap-icons.css">
  <style>
    .slider-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
    }
    .slider {
      width: 80%;
    }
    .slider-value {
      margin-top: 10px;
      font-size: 1.2em;
    }
    .image-container {
      margin-top: 20px;
      text-align: center;
    }
    .image-container img {
      max-width: 100%;
      height: auto;
    }
  </style>
</head>
<body>
<nav class="navbar navbar-light bg-light">
  <div class="container">
    <form class="form-inline my-2 my-lg-0 w-100 d-flex" action="/search" method="get">
      <input class="form-control flex-grow-1 mr-sm-2" type="search" name="q" placeholder="Search" aria-label="Search">
      <button class="btn btn-outline-secondary my-2 my-sm-0" type="submit">
        <i class="bi bi-search"></i>
      </button>
    </form>
  </div>
</nav>
{% block content %}

{% endblock %}

  <!-- Bootstrap and jQuery JS -->
  <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
  <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
  
</body>
</html>
"""


class StringLoader(BaseLoader):
    def get_source(self, environment, template):
        if template == "base_template":
            return base_template, None, lambda: True
        return None, None, None


app.jinja_env.loader = StringLoader()


@app.route("/")
def timeline():
    # connect to db
    timestamps = get_timestamps()
    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
{% if timestamps|length > 0 %}
  <div class="container">
    <div class="slider-container">
      <input type="range" class="slider custom-range" id="discreteSlider" min="0" max="{{timestamps|length - 1}}" step="1" value="{{timestamps|length - 1}}">
      <div class="slider-value" id="sliderValue">{{timestamps[0] | timestamp_to_human_readable }}</div>
    </div>
    <div class="image-container">
      <img id="timestampImage" src="/static/{{timestamps[0]}}.webp" alt="Image for timestamp">
    </div>
  </div>
  <script>
    const timestamps = {{ timestamps|tojson }};
    const slider = document.getElementById('discreteSlider');
    const sliderValue = document.getElementById('sliderValue');
    const timestampImage = document.getElementById('timestampImage');

    slider.addEventListener('input', function() {
      const reversedIndex = timestamps.length - 1 - slider.value;
      const timestamp = timestamps[reversedIndex];
      sliderValue.textContent = new Date(timestamp * 1000).toLocaleString();  // Convert to human-readable format
      timestampImage.src = `/static/${timestamp}.webp`;
    });

    // Initialize the slider with a default value
    slider.value = timestamps.length - 1;
    sliderValue.textContent = new Date(timestamps[0] * 1000).toLocaleString();  // Convert to human-readable format
    timestampImage.src = `/static/${timestamps[0]}.webp`;
  </script>
{% else %}
  <div class="container">
      <div class="alert alert-info" role="alert">
          Nothing recorded yet, wait a few seconds.
      </div>
  </div>
{% endif %}
{% endblock %}
""",
        timestamps=timestamps,
    )


@app.route("/search")
def search():
    q = request.args.get("q")
    entries = get_all_entries()
    embeddings = [np.frombuffer(entry.embedding, dtype=np.float32) for entry in entries]
    query_embedding = get_embedding(q)
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    indices = np.argsort(similarities)[::-1]
    sorted_entries = [entries[i] for i in indices]

    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
    <div class="container">
        <div class="row">
            {% for entry in entries %}
                <div class="col-md-3 mb-4">
                    <div class="card">
                        <a href="#" data-toggle="modal" data-target="#modal-{{ loop.index0 }}">
                            <img src="/static/{{ entry['timestamp'] }}.webp" alt="Image" class="card-img-top">
                        </a>
                    </div>
                </div>
                <div class="modal fade" id="modal-{{ loop.index0 }}" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
                    <div class="modal-dialog modal-xl" role="document" style="max-width: none; width: 100vw; height: 100vh; padding: 20px;">
                        <div class="modal-content" style="height: calc(100vh - 40px); width: calc(100vw - 40px); padding: 0;">
                            <div class="modal-body" style="padding: 0;">
                                <img src="/static/{{ entry['timestamp'] }}.webp" alt="Image" style="width: 100%; height: 100%; object-fit: contain; margin: 0 auto;">
                            </div>
                        </div>
                    </div>
                </div>
            {% endfor %}
        </div>
    </div>
{% endblock %}
""",
        entries=sorted_entries,
    )


@app.route("/entry/<int:entry_id>")
def entry_detail(entry_id: int):
    entry = get_entry_by_id(entry_id)
    if entry is None:
        abort(404)
    capture_filenames = _capture_filenames_for_timestamp(entry.timestamp)
    primary_capture = capture_filenames[0] if capture_filenames else None
    display_text = entry.normalized_text or entry.text or ""
    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
  <div class="container py-4">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <div>
        <h1 class="h3 mb-1">OpenRecall Entry {{ entry.id }}</h1>
        <div class="text-muted">{{ entry.timestamp | timestamp_to_human_readable }}</div>
      </div>
      <div>
        <a class="btn btn-outline-secondary btn-sm" href="/">Timeline</a>
      </div>
    </div>
    <div class="row">
      <div class="col-lg-8 mb-3">
        {% if primary_capture %}
          <img src="/static/{{ primary_capture }}" alt="OpenRecall capture {{ entry.id }}" class="img-fluid border rounded">
          {% if capture_filenames|length > 1 %}
            <div class="mt-2 small text-muted">
              Additional captures for this timestamp:
              {% for filename in capture_filenames[1:] %}
                <code>{{ filename }}</code>{% if not loop.last %}, {% endif %}
              {% endfor %}
            </div>
          {% endif %}
        {% else %}
          <div class="alert alert-warning">No screenshot file was found for this entry timestamp.</div>
        {% endif %}
      </div>
      <div class="col-lg-4">
        <div class="card mb-3">
          <div class="card-body">
            <h2 class="h5">Metadata</h2>
            <dl class="mb-0">
              <dt>ID</dt><dd><code>{{ entry.id }}</code></dd>
              <dt>Captured date</dt><dd><code>{{ entry.captured_date }}</code></dd>
              <dt>Timestamp</dt><dd><code>{{ entry.timestamp }}</code></dd>
              <dt>Application</dt><dd>{{ entry.app or "unknown" }}</dd>
              <dt>Window title</dt><dd>{{ entry.title or "(none)" }}</dd>
              <dt>Normalization</dt><dd>{{ entry.normalization_version or "(raw only)" }}</dd>
            </dl>
          </div>
        </div>
        <div class="card">
          <div class="card-body">
            <h2 class="h5">Text</h2>
            {% if display_text %}
              <pre style="white-space: pre-wrap; word-break: break-word;">{{ display_text }}</pre>
            {% else %}
              <div class="text-muted">No OCR text stored for this entry.</div>
            {% endif %}
          </div>
        </div>
      </div>
    </div>
  </div>
{% endblock %}
""",
        entry=entry,
        capture_filenames=capture_filenames,
        primary_capture=primary_capture,
        display_text=display_text,
    )


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(screenshots_path, filename)


if __name__ == "__main__":
    create_db()

    print(f"Appdata folder: {appdata_folder}")

    # Start the thread to record screenshots
    t = Thread(target=record_screenshots_thread)
    t.start()

    app.run(port=8082)
