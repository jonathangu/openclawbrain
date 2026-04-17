# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-031`
- winner mode: `graph_prior_only`
- trace hash: `sha256-7523c240f6bfd6f671f735664700ac6dc9001a24ca15ade4db3948ee85bd8854`
- fixture hash: `sha256-cfb2a9ca78d96cf92b242a10f940aec587e9ef8a10abc64c8e738df9cb79ae77`
- score hash: `sha256-a0d94340725a1d507601df58dc00dea35d88e63a78fe101973817a23af60e041`
- bundle hash: `sha256-acc3f0e071e7659cc482d9889ab387a34e1916b66e643f69e44e676756599ab4`

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6d2a784857e1edb0916a87444c502c8103349e39bca5b563e685c5f91e056e74 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a3ce049887225ef14182b22aea9d19841f66bbafc7af0a01f914ea6516a29e55 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f9f531123d4a660cf6504f4488393f02821ff595293f0fa770e9c7ee045881b7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e81aea66 | sha256-c0891e6a85f58352fbda055e4357b9894235cf7306713e42ea809c9c472f021a |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e81aea66 | sha256-a039df7f1b0e1d2715468f388d4580a026b5507d413edee8590cfa14fe3c405b |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-562dc819 | sha256-9e7988381947ffcc996827e32af275f24ec6653d05fd1e161982ce1ec3f47796 |
