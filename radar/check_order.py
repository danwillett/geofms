import zarr
import numpy as np

store = zarr.open('radar/outputs/dualpol_500m_2022-01-01_2026-04-04.zarr', mode='r')

# Compare time values accessible to each field
time_all  = store['time'][:]          # 9838 entries
time_kdp  = time_all[:9837]           # what KDP covers
time_rest = time_all[:9837]           # what truncated reflectivity covers

# If same object, they're aligned — now check the LAST KDP value
# If KDP's last entry matches the same timestamp as reflectivity's 9837th,
# they're in sync and only the very last scan is missing KDP.
print("Last time step in truncated set:     ", time_kdp[-1])
print("9838th time step (dropped):          ", time_all[9837])

# Also spot-check middle alignment
mid = 5000
print(f"\nreflectivity time[{mid}]: ", time_all[mid])
# The only way to check KDP alignment is by checking whether the timestamps
# stored in time[] are monotonically increasing
diffs = np.diff(time_all)
print(f"\nAny out-of-order timestamps? {(diffs < 0).sum()} inversions")
print(f"Min time gap (ns): {diffs.min()}")