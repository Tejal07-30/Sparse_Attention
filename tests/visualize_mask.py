import matplotlib.pyplot as plt
from attention.sliding_window import sliding_window_mask

mask = sliding_window_mask(32,4)

plt.imshow(~mask, cmap="Blues")
plt.title("Sliding Window Attention")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.colorbar(label="Allowed")
plt.show()