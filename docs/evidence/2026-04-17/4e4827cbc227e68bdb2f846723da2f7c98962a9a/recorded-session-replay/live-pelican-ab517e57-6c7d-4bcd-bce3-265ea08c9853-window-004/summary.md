# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ab517e57-6c7d-4bcd-bce3-265ea08c9853-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e0ad9ec13f7d5b82b36a685375b9d3d24391406d595ab3f8c2b0e0a5247f79c9`
- fixture hash: `sha256-0d1840771e0444519c0d4b5e3c3b57cee2fa58fe3cd78cd2a661af1ba4273a98`
- score hash: `sha256-b29d6e3046277f18b43cd89fe822812a9bd33a86bd537f4350413c6501edf422`
- bundle hash: `sha256-8fe5034269a0a39275dd1ffd337735850e0dfdec1fb0ec45c0a1e194483e7df8`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-24e5432aabba4b367a9ab9972174d2db006f79b43849cb63eacaea39404c4061 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-88e0802072d2331f0659cffba661be2703eb0f8b6c7a9e29bd870bbe50b11f5d |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4213c1edadc464d2bf71ae20082eaee666e438f4e58c4b72484cdf4ab9a9ce34 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-bbd3e809cb80200d874ebc16f8696b637e8a19ec449123d00350b84f499c1316 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-63e8e9b0 | sha256-7dfcea29fab1a59a72a1bb2b0336dbfdb1c96dabe044e34d74b686009ea249d8 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-63e8e9b0 | sha256-31fcebe8c17f57d0ddbb6267a7e964f84c4b8add28c9963cef21c3c3fbf00637 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-2d091baf | sha256-5a3180e1ac51aabb746d8706b07f0580257750a4d1ebf7a36dd52db25f48bbea |
