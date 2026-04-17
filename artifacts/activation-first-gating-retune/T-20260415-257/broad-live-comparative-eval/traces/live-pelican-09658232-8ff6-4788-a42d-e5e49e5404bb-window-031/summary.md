# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7523c240f6bfd6f671f735664700ac6dc9001a24ca15ade4db3948ee85bd8854`
- fixture hash: `sha256-cfb2a9ca78d96cf92b242a10f940aec587e9ef8a10abc64c8e738df9cb79ae77`
- score hash: `sha256-32703d98c7d5286338975aa7275d67dc909f046da8cec91845ff3f68034a0678`
- bundle hash: `sha256-e168ba695985d292131f4925d8d2c7feac1a243e3c058a9ca81000908e4e1989`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-af4b2f2d94a2b7dfe0347a004eb7b95f8c73a6da1a1ed505fb0a444fe4869d4c |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-015f208cc433e6c8f0587f5f5a354e59cbeb709e3d314414276461b6c4ce0d8f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-585940a522b94ae21b62601e2a8e62a300d5f3dc432e42633902d55957ccaace |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-47784cb1 | sha256-4d181392b79c71cb29110be469352f25fca483e11408cb678a01650c9aa2c1f0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-47784cb1 | sha256-9139f0e009dabaf816554327149bb1afc07e61cc1e0b64f3a0990879fd79693d |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-b58b2a64 | sha256-67c934525f7064a8b6264fb9b59cde6111216fa2f11b208cf4f4d33186e2041e |
