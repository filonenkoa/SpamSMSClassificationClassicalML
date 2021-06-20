import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the dataset.')
    parser.add_argument('--dataset', metavar='S', type=str,
                        help='Path to the dataset')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()

if __name__ == "__main__":
