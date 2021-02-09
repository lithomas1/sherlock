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

<main>
  <h1>Sherlock demo</h1>
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
</main>

<style>
  main {
    text-align: center;
    padding: 1em;
    max-width: 240px;
    margin: 0 auto;
  }

  h1 {
    color: #ff3e00;
    font-size: 4em;
    font-weight: 100;
  }

  @media (min-width: 640px) {
    main {
      max-width: none;
    }
  }
</style>
