# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31afb2df9c1a17ca25197bd8dab4006e37b9c5d5cee2757703f3aa5a6af3cc63`
- fixture hash: `sha256-998f6618e36f06829cb18a9eae15dbb334b923e47c420cfa28a2642db4d68155`
- score hash: `sha256-641e4a27870e276557a959599359bd45a81ddf4091d183f33f53291bc356ba56`
- bundle hash: `sha256-3b837e77f1904353c6755dd3932a9619deb5a611cb90e49dd4637e4b094e16aa`

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
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c0657d7b41d16e0a76bed7b5e5dcdadf4310444b0556eb5e7411f6141dac5dd0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-7df061ebf0a3a1a14abe1eadcbd89498b24c463a6b42f918d5cd3c0605f4a261 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f28752756277fa15459890b9f9e41d7ffb5883bbceb7bca3f03191c250ba63c9 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-0b0c341f84f7ab05c2cbd0d9673cd693012610dcd3d7e2eb59df6d8de3a430ae |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ebbd2743 | sha256-aabaf410c21fecdd48593ebe456bdb590b3f5487c0ee24a43a618acfcd090290 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-ebbd2743 | sha256-af1fec98b1e147f022fbbbac37867e23066b7dd084d1d55f48fb0fd282962d4c |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7d5c43ea | sha256-847ab796d7d5eaebd91178b8489ada52ecec0f79989e6560e13541b3eeb6a289 |
