# wwiznet_ppic (example)

Static web build (graph viewer) plus a tiny Python static file server for local preview.

quickstart: `python server.py`

## Data

example dataset is in data/data/
* nodes.tsv
* edges.tsv
* config.json - visualisation presets/nodeset presets etc


### nodes.tsv
* id column (required)
* arbitrary number of extra columns

### edges.tsv
* source (required, same id as in nodes.tsv id)
* target (required, same id as in nodes.tsv id)

### special column types/names
* x_&lt;layout&gt;, y_&lt;layout&gt;. z_&lt;layout&gt; - precomputed layouts, layout dropdown will be populated with &lt;layout&gt;'s
* *_a - array columns, comma seperated list of strings (eg: annotations)
* *_kv - comma seperated key value pairs (eg: key1:xyz,keyx2:asfd, ---)
* *_n - force column to be interpreted as numeric, eg metrics like degree, etc
* *_c - force column to interpreted as categorical, eg clusternumber, classification string etc
* *_c_kv - categorical keyvalue
* hl3d - optional, for molecular 3d viewer highlighted sites, comma seperated numerical index



### config.json
example preset option, can be generated with your settings in the app
* change settings in Controls menu etc
* save a preset (in Presets folder chose a name and press Save Current Presets)
* export json file (menu right top, Controls -> Presets -> Export local storage)



## Requirements

- Python 3.7+ (stdlib only; no `pip install`)

## Run locally

```bash
python server.py
```

Open [http://127.0.0.1:8888/](http://127.0.0.1:8888/) in your browser. Change the `PORT` value in `server.py` if 8888 is already in use.

Press **Ctrl+C** in the terminal to stop the server.

## Your data



| Path | Role |
|------|------|
| `index.html` | Entry page |
| `assets/` | Bundled JavaScript (ES modules) |
| `data/` | Dataset files (e.g. TSV, `config.json`) |
| `server.py` | Serves this folder over HTTP with correct `.js` MIME types on Windows |

## Notes

- The server always serves the directory that contains `server.py`, regardless of your current working directory.

