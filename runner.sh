docker build --tag task4 .
docker run --init --rm -u $(id -u):$(id -g) -v "$( cd "$( dirname "$0" )" && pwd )":/results task4

echo "this was the 'main' run [2 layers, 256 units]."
