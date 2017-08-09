MNIST_IMAGE = "sonm/mnist-nn"
MNIST_DOCKERFILE = "mnist.Dockerfile"

mnist:
    docker build -t ${MNIST_IMAGE}:latest -f ${MNIST_DOCKERFILE} .