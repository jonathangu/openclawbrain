# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-062`
- winner mode: `graph_prior_only`
- trace hash: `sha256-68dd4afa8cf6968b41418bd460fed5641fe37e0c30a004be1adb6fd97d678410`
- fixture hash: `sha256-0984347c035679f491e5e5ce92160de0970752142af6bd7d0f80779707ccfa84`
- score hash: `sha256-3183aa262309141cfced5666118399056f3acc3ecaad9becce89f5eccbc5399b`
- bundle hash: `sha256-def6e862c4d8706783f22781e176c540884287a21874c322847e417543b54fde`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4e7043d44034a818b042ec107f761c7c9e4d805591027e32242d8b764dc9d866 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f174355a7c3e443c23c7bbca77eb7e13d8796d4f857b71791ab840f6307b0080 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-82364c2c8e412fe663da02f5b919227917ba28434d0ba312e2d4e9bd154fb4e9 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-b55addcd1e2133031a03c969282af037e3a2cc5ace05d4a6f900126865956edb |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4a0aa5a5 | sha256-216dca01fedbaec34693db6db1debc41fc92adfd7d3a9d5fbc10ba02fc2941ef |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-4a0aa5a5 | sha256-37701e5aa9d0f923323678cc9ad3ba737ffde0ab5fa91162be7b77991cabe8e4 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-58243d02 | sha256-12c18392bd0fb9bf08a4316ff75873005e7adaa6af1ca9d1b3fdce353fb495f4 |
