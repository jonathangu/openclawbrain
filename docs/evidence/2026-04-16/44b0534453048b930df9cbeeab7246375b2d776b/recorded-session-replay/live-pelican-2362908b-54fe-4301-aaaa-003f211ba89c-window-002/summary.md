# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-df541968ca52654e5efa48a1a6713bb4511f8366d389ef30e36174b0478a0f72`
- fixture hash: `sha256-10a1d9d424d59bf74d6edef2d25c3d9864b38e04e75b6ff4b28dfed92245cd1e`
- score hash: `sha256-54a7504be69e6761c1901ff0b6d0dcdd3fd9a07dae5e11f0f99dcd650a7dc339`
- bundle hash: `sha256-6b5b4d38531d9bbbd75e268ff08f3489af8523164da2bf79ca96544d54038646`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-fe456f8f9f99c14a6c26ae3cfe1240fa644752e272eded6c0df3fca37912d301 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-afe3c6a35ff0148cf8f672ab7cd38289eef13312b14e3c28df11d2145fe795ab |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-9f500b1716402322ab22459148d934461e5b41ee076f924003cfc7ba77b77f03 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-bf9d63391b6e8c1c294905a7d10118264a689a5c602ede13569b1f8de5b1c14e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3f17ec20 | sha256-3f58c8c0552911a35bc481ace60b045081e19386c6646518f8a7c77d0839dab0 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3f17ec20 | sha256-c741d07d6df5a55fc0f4ec0e1a253fdfa15f80696904ca8be35f7b539309da19 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-e696ced9 | sha256-4773b8fac55271390fc8c4c02d9be18bc74a14a0b00c6938fde9f5923976a5c5 |
