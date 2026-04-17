# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-010`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bd69cb2bd54df203880c8fab0fa4c855341f06ecfb9d6ec642144558419aa71a`
- fixture hash: `sha256-f29630fbd2f41b8d395fae06865eb7778e00433b1298788381332e0703a42702`
- score hash: `sha256-e224221154a060cd3a1fb73d2e37c1bffd3f7662c8f7d72dd7b73fb3ae685e5b`
- bundle hash: `sha256-e0a2399881d75d6397f26f3f189ea1410a05d9e37dee350d8fa029eee04db6d6`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-80e65d944c503b7cf482a4ac157c70bd9810fcdc3cd3dc77c36042f87f3356ea |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0e05434c48381a9e848713186757641494d8392fa80b13d9ba6100e832920071 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5f988378ae462bf097aac1ae4591f5440f06c4265f6be9f5fa458997b77def67 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-90b527507dada1c49d05f48e022c889744248c5ba331ebcad69cfa77a74a3625 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dc87eff3 | sha256-933f60caee5f9d28036446ed031bc9bf7b876dd32836d8ebe67c10f458701e7f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-dc87eff3 | sha256-f8e04e839c59b1ad6d339fabebf165d8e89ebecf0b3994f07569b3a9e5bfdc23 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-1796e1b6 | sha256-033062bbcad0cea0806fab534349ec8aa714e8d0163d5f5a1a47248ada17a491 |
