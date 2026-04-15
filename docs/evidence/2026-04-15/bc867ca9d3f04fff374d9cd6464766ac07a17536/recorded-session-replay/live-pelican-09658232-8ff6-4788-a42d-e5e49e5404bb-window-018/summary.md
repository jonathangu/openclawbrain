# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31afb2df9c1a17ca25197bd8dab4006e37b9c5d5cee2757703f3aa5a6af3cc63`
- fixture hash: `sha256-998f6618e36f06829cb18a9eae15dbb334b923e47c420cfa28a2642db4d68155`
- score hash: `sha256-a06810bbdab411dbb1dad78a7d48650808408aed9d8eb075a16bda1057bd6900`
- bundle hash: `sha256-a5b98358317cb007af025dba4bf5a325434bba4778bd6f46acf21e01b24acb73`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-c0657d7b41d16e0a76bed7b5e5dcdadf4310444b0556eb5e7411f6141dac5dd0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0ea9f5a4c6e03e340895992b748e46c150b76affef79064e0d3cffaf6905914e |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9e2f8d53a8ae0ea1d25708ba501da275c233f9234708fd3675bf73d9fb5e3e9b |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-841f5c9076f39e6eeae4057d1a7061a04724984439f0ee2b0d960277a168c35e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-10af0657 | sha256-ea139c3ded4d6dd5cc5baccfef9c3f6df463ae708b7018ed1156d28c7ae68865 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-10af0657 | sha256-7af38fdec4a5292bb219651a1c1db458d58185246cfcbb5fd79e1db0a525a98a |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-10af0657 | sha256-ea139c3ded4d6dd5cc5baccfef9c3f6df463ae708b7018ed1156d28c7ae68865 |
