"""Takes as arguments an evaluation file and an answers pickle. Evaluates accuracy
of VAE, GAN, and NLM models."""
import sys
import pickle

eval_f_name = sys.argv[1]
ans_f_name = sys.argv[2]

answers = pickle.load(open(ans_f_name, 'rb'))

num_gan_wins = 0  # how many times did each model fool the evaluator?
num_vae_wins = 0
num_nlm_wins = 0
num_gan_losses = 0  # how many times did each model not fool the evaluator?
num_vae_losses = 0
num_nlm_losses = 0
num_gan_draws = 0  # how many times did the model tie with the real sentence?
num_vae_draws = 0
num_nlm_draws = 0
for index, line in enumerate(open(eval_f_name)):
    entries = line.split(" | ")

    # If this entry does not have a label, ignore it
    if len(entries) < 3:
        continue
    else:
        left_s = entries[1]
        right_s = entries[2]
        usr_choice = entries[0]
        each_answer = answers[index]
        label = each_answer[0]
        gen_first = each_answer[1]

        result = None
        if usr_choice == 'left':
            if gen_first:
                result = 'win'
            else:
                result = 'loss'

        elif usr_choice == 'right':
            if not gen_first:
                result = 'win'
            else:
                result = 'loss'

        elif usr_choice == 'both':
            result = 'draw'

        if label == 'gan':
            if result == 'win':
                num_gan_wins += 1
            elif result == 'loss':
                num_gan_losses += 1
            elif result == 'draw':
                num_gan_draws += 1
            else:
                raise AssertionError('Wrong result')
        elif label == 'vae':
            if result == 'win':
                num_vae_wins += 1
            elif result == 'loss':
                num_vae_losses += 1
            elif result == 'draw':
                num_vae_draws += 1
            else:
                print(result)
                raise AssertionError('Wrong result')
        elif label == 'nlm':
            if result == 'win':
                num_nlm_wins += 1
            elif result == 'loss':
                num_nlm_losses += 1
            elif result == 'draw':
                num_nlm_draws += 1
            else:
                raise AssertionError('Wrong result')
        else:
            raise ValueError('Incorrect label')

print('GAN: %s wins, %s losses, %s draws' % (num_gan_wins, num_gan_losses, num_gan_draws))
print('VAE: %s wins, %s losses, %s draws' % (num_vae_wins, num_vae_losses, num_vae_draws))
print('NLM: %s wins, %s losses, %s draws' % (num_nlm_wins, num_nlm_losses, num_nlm_draws))


