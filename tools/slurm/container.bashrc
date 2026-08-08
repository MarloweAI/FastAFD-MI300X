# Interactive shell startup used only by tools/slurm/shell.sh.
# Keep the user's normal aliases and shell settings, then make the container
# context unmistakable without loading the FastAFD Python environment.
if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

# Interactive conveniences. These aliases do not affect non-interactive scripts.
alias h='history'
alias cp='cp -i'
alias mv='mv -i'
alias rm='rm -i'

fastafd_prompt_node="${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-gpu}}}"
PS1="(fastafd:${fastafd_prompt_node}) "'\u:\w\$ '
unset fastafd_prompt_node

fastafd_tools_dir=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
fastafd_env_file="$fastafd_tools_dir/env.sh"

# Give background servers a chance to run their Python/FastAPI shutdown hooks
# before Slurm tears down the job cgroup. run_col_rocm.sh and run_afd_rocm.sh use
# exec, so SIGTERM reaches the Python server directly rather than stopping at a
# launcher shell.
fastafd_shutdown_background_jobs() {
  local original_status=$?
  local deadline
  local pid
  local -a pids=()
  local -a alive=()

  mapfile -t pids < <(jobs -pr)
  if (( ${#pids[@]} == 0 )); then
    return "$original_status"
  fi

  echo
  echo "Stopping ${#pids[@]} FastAFD container background job(s) gracefully..."
  kill -TERM "${pids[@]}" 2>/dev/null || true
  deadline=$((SECONDS + 30))

  while (( SECONDS < deadline )); do
    alive=()
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive+=("$pid")
    done
    (( ${#alive[@]} == 0 )) && break
    sleep 1
  done

  if (( ${#alive[@]} > 0 )); then
    echo "Forcing ${#alive[@]} background job(s) to stop after 30 seconds."
    kill -KILL "${alive[@]}" 2>/dev/null || true
  fi
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  echo "Background shutdown complete."
  return "$original_status"
}

trap fastafd_shutdown_background_jobs EXIT
trap 'exit 143' HUP TERM

echo
echo "Inside the FastAFD container on ${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-a GPU node}}}."
echo "The Python environment is not loaded yet. Run:"
echo "  source $fastafd_env_file"
echo "Exiting this shell gracefully stops its background server jobs."
echo
unset fastafd_env_file fastafd_tools_dir
