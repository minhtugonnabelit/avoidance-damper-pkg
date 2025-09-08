# getting workspace name by bash argument
WORKSPACE=$1

# create helper for listing arguments on --help or -h
list_args() {
    for arg in "$@"; do
        echo "Argument: $arg"
    done
}

ln -s $(pwd)/avoidance_damper_pkg/ $WORKSPACE/src/

echo "Avoidance Damper package installed in $WORKSPACE/src/avoidance_damper_pkg"