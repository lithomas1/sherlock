<script>
  let isGettingUsername = true;
  let name = '';
  let claims = [];

  function selectUserOnKeydown(e) {
    if (e.key === 'Enter')
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
    {#if claim === 'loading...' || claim === 'no tweets found'}
      <p>{claim}</p>
    {:else}
      <div class="claim">
        <span class="claimtext">{claim}</span>
      </div>
    {/if}
  {/each}
{/if}

<style>
  .claim {
    margin: auto;
    transition: .4s ease-in-out;
    width: 80vw;
    padding: 1vh;
    padding-top: .5vh;
  }

  .claim:nth-of-type(odd) {
    background: #E5E5E5;
  }

  .claim:hover {
    background: lightblue;
    cursor: pointer;
    font-size: 1.2em;
    transition: .2s ease-in-out;
  }
</style>
