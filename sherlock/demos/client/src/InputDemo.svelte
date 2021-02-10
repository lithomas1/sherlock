<script>
  import CheckClaim from './CheckClaim.svelte';

  let claim = '';
  let claimToVerify = '';

  function claimOnKeydown(e) {
    if (e.key === 'Enter')
      claimVerification();
  }

  function claimVerification() {
    claimToVerify = claim;
  }
</script>

<input bind:value={claim} on:keydown={claimOnKeydown} />
<button on:click={claimVerification}>
  Verify claim
</button>

{#if claimToVerify !== ''}
  <div class="padding" />
  <CheckClaim claim={claimToVerify} />
{/if}

<style>
  .padding {
    min-height: 2.5%;
  }
</style>
