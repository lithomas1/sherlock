<script>
  let isGettingUsername = true;
  let name = '';
  let claims = [];

  function selectUserOnKeydown(e) {
    if (e.key == 'Enter')
      selectUser();
  }

  async function selectUser() {
    isGettingUsername = false;
    claims = ['loading...'];
    claims = await fetch(`/claims?username=${name}`)
      .then(r => r.json());
    if (claims.length == 0)
      claims = ['no tweets found'];
  }
</script>

{#if isGettingUsername}
  <input bind:value={name} on:keydown={selectUserOnKeydown} />
  <button on:click={selectUser}>
    Select user
  </button>
{:else}
  {#each claims as claim}
    <p>{claim}</p>
  {/each}
{/if}
