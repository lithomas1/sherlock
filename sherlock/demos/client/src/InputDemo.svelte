<script>
  let claim = '';
  let claimToVerify = '';

  function claimOnKeydown(e) {
    if (e.key === 'Enter')
      claimVerification();
  }

  async function claimVerification() {
    claimToVerify = claim;
  }
</script>

<input bind:value={claim} on:keydown={claimOnKeydown} />
<button on:click={claimVerification}>
  Verify claim
</button>

{#if claimToVerify !== ''}
  <p>{claimToVerify}</p>
{/if}

<style>
</style>
