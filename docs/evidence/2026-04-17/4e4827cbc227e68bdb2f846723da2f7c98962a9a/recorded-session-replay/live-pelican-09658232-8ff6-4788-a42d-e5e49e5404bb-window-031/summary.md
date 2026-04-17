# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7523c240f6bfd6f671f735664700ac6dc9001a24ca15ade4db3948ee85bd8854`
- fixture hash: `sha256-cfb2a9ca78d96cf92b242a10f940aec587e9ef8a10abc64c8e738df9cb79ae77`
- score hash: `sha256-984e9bcfd5504b9985d55643b3c482bbf86b752840d73d63456979f185bf805e`
- bundle hash: `sha256-69bb037db5af13a7fa104e89d85295906feab35c4915b84a5d1a8d60564129e6`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e58229df13dcf750a6ed3af7bdcf953c6815a1f652d19ad1292758f1bf838de0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5643576ebe332a21259c3188442e5ba9a63424cf591701e75bdaa4a53ad16274 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-54203716e93785cf547c34ee5d6072d66e0ff08e3f549a9c5d673bbe3893962c |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-e58c7d535acb28a1a0330de586ec7372a512cfb8f771f32ff4da8d1dcdc085b8 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-110fe83d | sha256-4d34f8b2feefb5c0bccc19bf3f0aa3e8e11f63371b9d811cda993d7654e76c2c |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-110fe83d | sha256-c97cee946ed6b779c4fcb3c367781f7fd18f4057e5a6056d7dae114be241d4f9 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7f22c5f0 | sha256-33d56da5d95e1d29a961690cdeea21cae55fb280a7f7d818c6628be984fab648 |
