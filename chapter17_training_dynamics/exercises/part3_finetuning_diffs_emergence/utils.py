"""Plot display only; all scientific methods live in the master."""
import matplotlib.pyplot as plt

def finish_figure(fig):
    """Apply layout and show an already constructed figure."""
    fig.tight_layout()
    plt.show()
