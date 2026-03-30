#!/bin/bash

CONDA_ENV_LIB_PATH="$CONDA_PREFIX/lib"
LIBFFI_SO_PATH="$CONDA_ENV_LIB_PATH/libffi.so.7"
BACKUP_PATH="$CONDA_ENV_LIB_PATH/libffi_bak.so.7"
NEW_LINK_TARGET="/lib/x86_64-linux-gnu/libffi.so.7.1.0"

if [ ! -e "$LIBFFI_SO_PATH" ]; then
    echo "$LIBFFI_SO_PATH does not exist."
    exit 1
fi

echo "Current libffi.so.7 link:"
ls -l "$LIBFFI_SO_PATH"

mv "$LIBFFI_SO_PATH" "$BACKUP_PATH"
echo "Backup created: $BACKUP_PATH"

ln -s "$NEW_LINK_TARGET" "$LIBFFI_SO_PATH"
if [ $? -eq 0 ]; then
    echo "Created new link: $LIBFFI_SO_PATH -> $NEW_LINK_TARGET"
else
    echo "Error creating new link."
    exit 1
fi

sudo ldconfig
if [ $? -eq 0 ]; then
    echo "Updated dynamic linker cache."
else
    echo "Error updating ldconfig."
    exit 1
fi
