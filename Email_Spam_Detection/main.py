from src.predict import predict_sms

print("=" * 60)
print("        EMAIL SPAM DETECTION SYSTEM")
print("=" * 60)

message = input("\nEnter Email / SMS:\n\n")

prediction, spam_probability, ham_probability = predict_sms(message)

print("\n" + "=" * 60)

if prediction == "SPAM":

    print("🚨 RESULT : SPAM MESSAGE")

else:

    print("✅ RESULT : GENUINE MESSAGE")

print()

print(f"Spam Probability : {spam_probability*100:.2f}%")

print(f"Ham Probability  : {ham_probability*100:.2f}%")

confidence = max(

    spam_probability,

    ham_probability

)

print(f"Confidence       : {confidence*100:.2f}%")

print("=" * 60)