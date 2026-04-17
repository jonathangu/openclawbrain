# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-046`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3c2d2ba443dbef189a04f697781c21859dae784757f070f6d624a5c22c1fd87e`
- fixture hash: `sha256-d5090234178376a892e6c521a05dfe5104bf688b9e6c7c68cfaf8797d0e0e324`
- score hash: `sha256-49249f4d555ffcf51d7f3454b03db81fd9389fa54e03b978ea75db32744261a9`
- bundle hash: `sha256-04daf5ce2ea691d6a5f57f37d520e46c4ebf1316eed4709160ae95c045077f05`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-30cb0b5562af7757589ca1411482395b52af039eced1208652e3e0610a2b0728 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-03caf6c7cb73e8cbc46d328b8ee923c00d33d3ffeae9f30a748c6812599741e1 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3a0f9632ea63a6ea1250bf158101bc650b60c0e30e40c49c3314f0af6a132a33 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-aa4fe753813aef59fc27b1f51422b1afccbc98b18b63835e2fca3034c11a98ad |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3fe101e4 | sha256-a77957de9ed1774abc5c72a71d994c0f904865a691f382d68e4fa2ae4ca92b75 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-3fe101e4 | sha256-4603d5827b0e58530101ee8e6e9796ef54b036e1457e072850ef0996499fb4aa |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-31d1beaf | sha256-2a8764d53be7f8b69a3a43282685373433e1687092072247685acd4356309b20 |
