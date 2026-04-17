# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-64a26e8f5e6980d9246d302acab2d34c4e18ddd7be07096a6ca889aa90e2228a`
- fixture hash: `sha256-833c77206f16af416cd188d9e8ee18c5e59708b98a4500bfd6d7d22e62fa078a`
- score hash: `sha256-9bd24e18c59ee60c1a0ab870d072e98abc2de7b51f9215727d01dbfa7e64c785`
- bundle hash: `sha256-43d59b9c1a5087ee47b45d669e40ddc10fd3fb66e20c8b46dc43beaee5831ad4`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-947e512b0a82ec6d517ce602229a8e508d29ae58b836a4631a42b14c828dead3 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-188561717068dcd5e7e486f7f02bddf572865a5caa0cc9a9dffa4be45ffa4007 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-5ed502793a641d81b5775601a2891e02ffaed32ed2fa6c089c6eed95626bc30f |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-a095e8ec9304e7d918bc3b3401febf896c2d4de25d0fdead5fe74d0b305a6946 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-35f003fb | sha256-79c461c0b8638ce5d22c1b20c42744ad65ae6ecc2de59c4f64ea33d354e3c97e |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-35f003fb | sha256-3932c4becb2f7804193161e2cb551d3260875de9af425eaa77acdfc3c0e8d9bf |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-30ec4fb2 | sha256-eb9db6171bcc2e42860c359aab1e061d71eeb834b7bb3f0701ef9441354f574d |
