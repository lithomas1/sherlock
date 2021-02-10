<script>
  import CheckClaim from './CheckClaim.svelte';

  let name = '';
  let claims = [];
  let claimToVerify = '';

  function selectUserOnKeydown(e) {
    if (e.key === 'Enter')
      selectUser();
  }

  async function selectUser() {
    claims = ['loading...'];
    claims = await fetch(`/claims?username=${name}`)
      .then(r => r.json());
    if (claims.length == 0)
      claims = ['no tweets found'];
  }
</script>

<input bind:value={name} on:keydown={selectUserOnKeydown} />
<button on:click={selectUser}>
  Select user
</button>
{#each claims as claim}
  {#if claim === 'loading...' || claim === 'no tweets found'}
    <p>{claim}</p>
  {:else}
    <div class="claim" on:click={() => claimToVerify = claim}>
      <span class="claimtext">{claim}</span>
    </div>
  {/if}
{/each}

<div
  class="modal"
  style={claimToVerify ? 'display:block;' : ''}
  on:click={() => claimToVerify = ''}>
  {#if claimToVerify !== ''}
    <div on:click={(e) => e.stopPropagation()} class="modal-content">
      <span on:click={() => claimToVerify = ''} class="close">
        &times;
      </span>
      <p><q>{claimToVerify}</q></p>
      <CheckClaim claim={claimToVerify} />
    </div>
  {/if}
</div>

<style>
  .claim {
    margin: auto;
    transition: .4s ease-in-out;
    width: 80vw;
    padding: 1vh;
    padding-top: .5vh;
  }

  .claim:nth-child(even) {
    background: #E5E5E5;
  }

  .claim:hover {
    background: lightblue;
    cursor: pointer;
    font-size: 1.2em;
    transition: .2s ease-in-out;
  }

  .modal {
    display: none;
    position: fixed;
    z-index: 1;
    margin: auto;
    background-color: rgba(0, 0, 0, .4);
    top: 0;
    left: 0;
    padding-top: 10%;
    width: 100%;
    height: 100%;
  }

  .modal-content {
    background-color: #FEFEFE;
    margin: auto;
    padding: 20px;
    width: 50%;
    border: 1px solid #888;
  }

  .close {
    color: #AAAAAA;
    float: right;
    font-size: 1.8em;
    font-weight: bold;
    transition: .2s ease-in-out;
  }

  .close:hover {
    color: #000000;
    text-decoration: none;
    cursor: pointer;
    transform: scale(1.05);
  }
</style>
