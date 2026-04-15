# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7523c240f6bfd6f671f735664700ac6dc9001a24ca15ade4db3948ee85bd8854`
- fixture hash: `sha256-cfb2a9ca78d96cf92b242a10f940aec587e9ef8a10abc64c8e738df9cb79ae77`
- score hash: `sha256-1efd9ad501576c851ba8df81de1d2837782dd67fff29dea3485530d77e1dec0a`
- bundle hash: `sha256-337690b9c9eba8700e33f25aee306999c395cc2f25a16771edd6cf2411d39b7e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-e58229df13dcf750a6ed3af7bdcf953c6815a1f652d19ad1292758f1bf838de0 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-422e666ff7a716cadf4ae1344fcd72b0f1dd74230c3b748baed1d12ecfbe99ed |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8c30ea465a0cfa5d5033dfb4d1b62682086dc937a21b7cd22f313484dc5aeee0 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-6e15f612241887e90c20754c7e22484b1f4798196c16392c32921d884b575af4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b041765b | sha256-37e5ec929a31ea8a65c217b55234e554d42c004dd3896559887b7ee874505839 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b041765b | sha256-78ca3b6a23ddda5357c1b3a69d2591e1c4108ef5e6ed5cede8fcd26c4e0b3ebd |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b041765b | sha256-37e5ec929a31ea8a65c217b55234e554d42c004dd3896559887b7ee874505839 |
